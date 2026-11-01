"""
Shear-driven (Kelvin–Helmholtz) interior mixing of Jackson, Hallberg & Legg
(2008, "JHL"), column version ported from MOM6 ``kappa_shear`` (Jackson task 1).

This module is the **NumPy oracle**: a plain-Python, one-column port that keeps
MOM6's iteration structure step by step, so that it can be compared with the
compiled MOM6 routine (task 2) and later with a Numba kernel (task 3).  Speed is
irrelevant here.

Reference implementation (cited as ``file:line``, MOM6 git 0e9a561):
``MOM_kappa_shear.F90`` = ``~/pyOCEAN_refs/MOM6/src/parameterizations/vertical/MOM_kappa_shear.F90``
(``kappa_shear_column`` :866-1374, ``calculate_projected_state`` :1379-1506,
``find_kappa_tke`` :1701-2041; reference read in
``docs/design/shear_refs/mom6_kappa_shear.md``).

Paper: Jackson, L., Hallberg, R. & Legg, S. (2008), A parameterization of
shear-driven turbulence for ocean climate models, J. Phys. Oceanogr. 38,
1033–1053, https://doi.org/10.1175/2007JPO3779.1.  The time-step control
follows Hallberg (2000) [verificar].

Physics (interface quantities: diffusivity κ [m² s⁻¹], turbulent kinetic energy
per unit mass Q [m² s⁻²], N² and S² [s⁻²])
------------------------------------------------------------------------
1. **Source of κ** (right-hand side of JHL eq. 10, with F(Ri) of JHL §4a,
   which is unnumbered; MOM_kappa_shear.F90:1646-1652), local
   gradient Richardson number Ri = N²/S²:

       K_src = 2 γ |S| (Ri_c S² − N²) / (Ri_c S² + F_c N²)   if N² < Ri_c S²,
       K_src = 0                                            otherwise  [s⁻¹],

   i.e. 2γ|S| F(Ri), F = (Ri_c − Ri)/(Ri_c + F_c Ri), with γ = ``shearmix_rate``,
   Ri_c = ``ri_crit``, F_c = ``fri_curvature``.
2. **κ equation** (JHL eq. 10; :1775-1823), elliptic with κ = 0 at the surface
   and the bed:

       κ / L_d² − ∂²κ/∂z² = K_src,
       1/L_d² = (N²/λ² + f²)/Q + 1/L_bdry²,   λ = ``kappa_buoy_scale``,
       1/L_bdry² = (1/d_top + 1/d_bot)² / LZ_RESCALE²   (:1039-1049),

   d_top, d_bot = distances [m] from the interface to the surface and the bed.
3. **Steady TKE equation** (JHL eq. 11 with MOM6's decay; :1709-1773), Q = q0 at
   the surface and Q = Q_min at the bed:

       0 = ∂_z[(κ + κ0) ∂_z Q] + (κ + κ0) S² − κ N² − (Q − q0) √(c_N² N² + c_S² S²),

   c_N = ``tke_n_decay``, c_S = ``tke_shear_decay``, q0 = ``tke_background``,
   Q_min = max(q0, 1e-20 m² s⁻²) (:1632).  The buoyancy sink κN² is linearised as
   Q (κ/Q) N² with K_Q = κ/Q from the previous iterate (:1733).
4. **Coupled solve** (``_find_kappa_tke``, :1701-2041): alternating Q and κ
   tridiagonal solves (Picard) that switch to a Newton solve of the coupled
   system once the relative changes fall below ``Newton_err`` (0.2, halved after
   an aborted Newton step), at most ``max_rino_it`` iterations, converged when
   |Δκ| ≤ ``tol_err`` (κ0 + κ) everywhere.  κ below ``kappa_trunc`` is set to 0
   and κ between 1× and 2× ``kappa_trunc`` is ramped as 2(κ − κ_trunc).
   The corrected Newton derivative is used (``KAPPA_SHEAR_ITER_BUG`` = False:
   dκ/dQ with dz_Int, :1860-1866).  As written in MOM6 (:2003-2021), the first
   iteration of a solve aborts Newton (and halves ``Newton_err`` to ``tol_err``)
   unless every interface changes by ≤ 20 %; an interface with κ = 0 restarts at
   Q_min (its Q then changes by orders of magnitude), so in practice Newton is
   reached only in warm-started solves of a column turbulent on every interior
   interface (task 2: matched against compiled MOM6, see
   ``validation/mom6_column/README.md``).
5. **Integration over Δt** (``kappa_shear_column``, :1052-1374):
   * one implicit step of κ0 diffusion of u, v, b over Δt before any gradient
     (:1052-1088; no-slip u, v at the bed, insulating b); κ0 is **not** part of
     the returned κ;
   * interface cells h_Int from the harmonic-mean split of each layer (:1091-1109);
   * up to ``max_ks_it`` sub-steps: κ from the current state, then the longest
     sub-step dt_now ≤ dt_rem over which K_src stays within
     [K_src − 0.95 L, K_src + ``max_kap_src_chg`` L] (L = local source) or within
     2 ``tol_err`` × the accumulated source (halving, then 5 bisection
     refinements, :1218-1296); predictor–corrector κ_mid = ½(κ_out + κ_pred)
     (:1316-1357); the column state (u, v, b — inside this routine only) is mixed
     with κ_mid over dt_now (:1361-1368);
   * the returned κ and Q are the **Δt-averages** Σ κ_mid dt_now / Δt.
   Each call seeds κ = ``kd_seed`` (:325); nothing is carried between calls.
6. **N²** = max(0, Δb/Δz) at interfaces (:1499): no convective mixing here.

Decisions of this port (Claude, 2026-10-30; see the task-1 report)
------------------------------------------------------------------
* **Buoyancy as one tracer.**  MOM6 carries T and S and multiplies their
  interface differences by ∂b/∂T, ∂b/∂S from the EOS at the interface T, S and
  pressure (:1121-1147).  Here the caller passes b = −g(ρ − ρ₀)/ρ₀ [m s⁻²] of each
  layer and b is mixed as the only tracer, N² = (b_above − b_below)/Δz.  This is
  exact for a linear EOS in T only (DOME, Ilicak, the task-2 comparison, where
  MOM6's ∂b/∂T is constant), and the local linearisation for a nonlinear EOS
  (TEOS-10): the mixing inside the solver then moves b instead of (T, S).
* Boussinesq, heights in m (H = Z, dz_h_Int = 1); no surface pressure (it only
  enters the EOS derivatives).  f² is one scalar per column.
* MOM6 options not exposed in ``JacksonParams`` take MOM6's defaults:
  tracer-point mode (``VERTEX_SHEAR`` = False), ``USE_RESTRICTIVE_TOLERANCE_CHECK``
  = False, ``KAPPA_SHEAR_ALL_LAYER_TKE_BUG`` = False (Q returned is the
  Δt-average), ``VEL_UNDERFLOW`` = 0 (no velocity is zeroed), no-flux TKE
  boundary conditions off (hard-wired off in MOM6, :1608-1609),
  ``KAPPA_SHEAR_ITER_BUG`` = False.
* ``kappa_shear_column`` does no thin-layer merging: every layer must be
  thicker than zero.  ``kappa_shear_columns`` (Numba, threads over columns) and
  its oracle ``kappa_shear_columns_numpy`` merge thin layers first, as MOM6
  ``KAPPA_SHEAR_ELIM_MASSLESS`` (:254-303) with pyOCEAN's relative threshold
  (task 3; section at the end of this module).

Conventions of this module
--------------------------
* ``kappa_shear_column`` takes model arrays: layers k = 0 (bed) … N−1 (surface),
  interfaces 0 (bed) … N (surface).
* Internally the arrays follow MOM6: **1-based, top-down** (layer 1 = surface,
  interface 1 = surface, interface nz+1 = bed), length nz + 2 with index 0
  unused, so every line keeps the indices of the Fortran it ports.
* Fortran locals that persist across the iterations of one call (e.g. dK, dQ,
  I_Ld2, the N2/S2 arrays that ``calculate_projected_state`` updates only in its
  active range, and loop indices left after ``exit``) are kept with the same
  persistence; arrays that MOM6 leaves uninitialised start at zero.

Nonlinear EOS (TEOS-10 in the 3-D core, task 6)
-----------------------------------------------
MOM6 carries T and S through ``calculate_projected_state`` and forms N² from the fixed interface coefficients
dbuoy_dT, dbuoy_dS of the locally referenced density (MOM_kappa_shear.F90:1128-1136, 1496-1500).  This port carries
one buoyancy b (exact for the linear EOS, whose coefficients are constant).  With a nonlinear EOS the column drivers
(``kappa_shear_columns`` and its oracle, ``eos`` given) build a **locally referenced buoyancy** of the merged
column instead of −g(ρ − ρ₀)/ρ₀: b_0 = 0, b_q = b_q−1 + Δb_q with Δb_q MOM6's jump dbuoy_dT ΔT + dbuoy_dS ΔS at the
merged interface (``physics.eos.stratification.buoyancy_jump_elem``; merged T, S = Σ h·x / Σ h, p = Σ g ρ₀ hc
from the surface; ``_local_buoyancy``).  The initial N² of the column (the dt = 0 projection) is then MOM6's to
the round-off of the running sum; inside the sub-stepping the projected Δb of a diffused b differs from
dbuoy_dT ΔT′ + dbuoy_dS ΔS′ of the separately diffused T′, S′ by the change of the coefficients between interfaces
times the change of the gradients (second order; class (c), a port of the two-tracer projection is left for a
later task).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numba import njit, prange

from physics.eos.stratification import buoyancy_jump_elem, eos_code

# A negligible fractional change in TKE [nondim] (MOM_kappa_shear.F90:1605).
ROUNDOFF: float = 1.0e-16
# Floor of the TKE that can be solved for [m² s⁻²] (MOM_kappa_shear.F90:1632).
TKE_FLOOR: float = 1.0e-20
# Number of 2-fold refinements of the sub-step (MOM_kappa_shear.F90:1022).
DT_REFINEMENTS: int = 5


@dataclass(frozen=True)
class JacksonParams:
    """
    Parameters of the Jackson et al. (2008) scheme; defaults = MOM6's defaults,
    i.e. JHL best-fit set 1 (MOM_kappa_shear.F90:2106-2112).

    Attributes
    ----------
    ri_crit : float
        Critical Richardson number Ri_c [nondim]; RINO_CRIT 0.25 (:2159).
    shearmix_rate : float
        γ, the nondimensional rate of shear mixing [nondim]; SHEARMIX_RATE 0.089 (:2162).
    fri_curvature : float
        F_c, curvature of F(Ri) [nondim]; FRI_CURVATURE −0.97 (:2189).
    kappa_buoy_scale : float
        λ, scale of the buoyancy decay length of κ [nondim];
        KAPPA_BUOY_SCALE_COEF 0.82 (:2204).
    tke_n_decay : float
        c_N, TKE decay by stratification [nondim]; TKE_N_DECAY_CONST 0.24 (:2193).
    tke_shear_decay : float
        c_S, TKE decay by shear [nondim]; TKE_SHEAR_DECAY_CONST 0.14 (:2200).
    kd_seed : float
        First guess of κ at every call [m² s⁻¹]; KD_SEED_KAPPA_SHEAR 1.0 (:2179).
    kd_trunc_frac : float
        κ_trunc / κ0 [nondim]; KD_TRUNC_KAPPA_SHEAR = 0.01 KD_KAPPA_SHEAR_0 (:2184).
    tol_err : float
        Relative tolerance of the κ iteration [nondim]; KAPPA_SHEAR_TOL_ERR 0.1 (:2219).
    max_rino_it : int
        Maximum κ–Q iterations per solve; MAX_RINO_IT 50 (:2166).
    max_ks_it : int
        Maximum sub-steps over Δt; MAX_KAPPA_SHEAR_IT 13 (:2236).
    max_kap_src_chg : float
        Tolerated growth of K_src within a sub-step, relative to the local
        source [nondim]; KAPPA_SHEAR_MAX_KAP_SRC_CHG 10 (:2248).
    tke_background : float
        q0, background TKE [m² s⁻²]; TKE_BACKGROUND 0 (:2226).
    lz_rescale : float
        Rescaling of the boundary distance L_bdry [nondim]; LZ_RESCALE 1 (:2214).
    """

    ri_crit: float = 0.25            # RINO_CRIT
    shearmix_rate: float = 0.089     # SHEARMIX_RATE
    fri_curvature: float = -0.97     # FRI_CURVATURE
    kappa_buoy_scale: float = 0.82   # KAPPA_BUOY_SCALE_COEF
    tke_n_decay: float = 0.24        # TKE_N_DECAY_CONST
    tke_shear_decay: float = 0.14    # TKE_SHEAR_DECAY_CONST
    kd_seed: float = 1.0             # KD_SEED_KAPPA_SHEAR
    kd_trunc_frac: float = 0.01      # KD_TRUNC_KAPPA_SHEAR / KD_KAPPA_SHEAR_0
    tol_err: float = 0.1             # KAPPA_SHEAR_TOL_ERR
    max_rino_it: int = 50            # MAX_RINO_IT
    max_ks_it: int = 13              # MAX_KAPPA_SHEAR_IT
    max_kap_src_chg: float = 10.0    # KAPPA_SHEAR_MAX_KAP_SRC_CHG
    tke_background: float = 0.0      # TKE_BACKGROUND
    lz_rescale: float = 1.0          # LZ_RESCALE


def _k_source(n2: float, s2: float, p: JacksonParams) -> float:
    """
    Shear source of κ, K_src = 2γ|S| (Ri_c S² − N²)/(Ri_c S² + F_c N²) [s⁻¹]
    (JHL eq. 10 with F(Ri) of JHL §4a; MOM_kappa_shear.F90:1646-1652), for
    N² < Ri_c S² only.
    """
    rc = p.ri_crit
    return (2.0 * p.shearmix_rate * np.sqrt(s2)) * \
        ((rc * s2 - n2) / (rc * s2 + p.fri_curvature * n2))


def _projected_state(kappa: np.ndarray, u0: np.ndarray, v0: np.ndarray, b0: np.ndarray,
                     dt: float, nz: int, dz: np.ndarray, i_dz_int: np.ndarray,
                     u: np.ndarray, v: np.ndarray, b: np.ndarray,
                     n2: np.ndarray, s2: np.ndarray,
                     ks_int: int | None = None, ke_int: int | None = None) -> None:
    """
    State after mixing u, v, b with κ for dt, and its N², S² (in place).

    Port of ``calculate_projected_state`` (MOM_kappa_shear.F90:1379-1506): one
    backward-Euler step of ∂c/∂t = ∂_z(κ ∂_z c) on layers ks … ke only
    (ks = ks_int − 1, ke = ke_int), no-slip u, v at the bed when the mixing
    reaches it, insulating b; then

        S² = |Δu|² / Δz²,   N² = max(0, Δb/Δz),   Δz = ½(dz_{k−1} + dz_k)

    at the interfaces ks … ke+1 (unprojected values outside the range).
    Outside these ranges the output arrays keep their previous contents, as in
    MOM6 (intent(inout)).  All arrays 1-based, top-down; u, v [m s⁻¹],
    b [m s⁻²], κ [m² s⁻¹], dz [m], i_dz_int [m⁻¹], dt [s].
    """
    ks, ke = 1, nz
    if ks_int is not None:
        ks = max(ks_int - 1, 1)
    if ke_int is not None:
        ke = min(ke_int, nz)
    if ks > ke:
        return
    # The final update of kappa_shear_column passes the same arrays as input
    # and output (:1364-1366); copy so that reads of u0 stay unprojected.
    u0 = u0.copy()
    v0 = v0.copy()
    b0 = b0.copy()

    if dt > 0.0:
        c1 = np.zeros(nz + 2)
        a_b = dt * (kappa[ks + 1] * i_dz_int[ks + 1])
        b1 = 1.0 / (dz[ks] + a_b)
        c1[ks + 1] = a_b * b1
        d1 = dz[ks] * b1  # = 1 - c1
        u[ks] = (b1 * dz[ks]) * u0[ks]
        v[ks] = (b1 * dz[ks]) * v0[ks]
        b[ks] = (b1 * dz[ks]) * b0[ks]
        for k in range(ks + 1, ke):
            a_a = a_b
            a_b = dt * (kappa[k + 1] * i_dz_int[k + 1])
            bd1 = dz[k] + d1 * a_a
            b1 = 1.0 / (bd1 + a_b)
            c1[k + 1] = a_b * b1
            d1 = bd1 * b1
            u[k] = b1 * (dz[k] * u0[k] + a_a * u[k - 1])
            v[k] = b1 * (dz[k] * v0[k] + a_a * v[k - 1])
            b[k] = b1 * (dz[k] * b0[k] + a_a * b[k - 1])
        # Insulating b, or mixing simply stopping (:1447-1450).
        a_a = a_b
        b1 = 1.0 / (dz[ke] + d1 * a_a)
        b[ke] = b1 * (dz[ke] * b0[ke] + a_a * b[ke - 1])
        # No-slip u, v when the mixing reaches the bed (:1452-1463).
        if ke == nz:
            a_b = dt * (kappa[nz + 1] * i_dz_int[nz + 1])
            b1nz_0 = 1.0 / ((dz[nz] + d1 * a_a) + a_b)
        else:
            b1nz_0 = b1
        u[ke] = b1nz_0 * (dz[ke] * u0[ke] + a_a * u[ke - 1])
        v[ke] = b1nz_0 * (dz[ke] * v0[ke] + a_a * v[ke - 1])
        # VEL_UNDERFLOW = 0: no velocity is zeroed (:1464-1465, 1469-1470).
        for k in range(ke - 1, ks - 1, -1):
            u[k] = u[k] + c1[k + 1] * u[k + 1]
            v[k] = v[k] + c1[k + 1] * v[k + 1]
            b[k] = b[k] + c1[k + 1] * b[k + 1]
    else:
        u[1:nz + 1] = u0[1:nz + 1]
        v[1:nz + 1] = v0[1:nz + 1]
        b[1:nz + 1] = b0[1:nz + 1]

    # Squared shear at interfaces, S² = |Δu|²/Δz² (:1483-1491).  Squares are written x*x as gfortran compiles
    # Fortran x**2: NumPy/Python x**2 calls libm pow(), which differs from x*x by 1 ulp for ~0.1 % of the
    # arguments (task 3: this was the 2.6e-26 m² s⁻² left in Q against MOM6 in task 2; Numba compiles x**2 to x*x).
    s2[1] = 0.0
    s2[nz + 1] = 0.0
    if ks > 1:
        s2[ks] = ((u[ks] - u0[ks - 1]) * (u[ks] - u0[ks - 1]) + (v[ks] - v0[ks - 1]) * (v[ks] - v0[ks - 1])) \
            * (i_dz_int[ks] * i_dz_int[ks])
    for k in range(ks + 1, ke + 1):
        s2[k] = ((u[k] - u[k - 1]) * (u[k] - u[k - 1]) + (v[k] - v[k - 1]) * (v[k] - v[k - 1])) \
            * (i_dz_int[k] * i_dz_int[k])
    if ke < nz:
        s2[ke + 1] = ((u0[ke + 1] - u[ke]) * (u0[ke + 1] - u[ke]) + (v0[ke + 1] - v[ke]) * (v0[ke + 1] - v[ke])) \
            * (i_dz_int[ke + 1] * i_dz_int[ke + 1])

    # N² = max(0, (b_above − b_below)/Δz): no convection (:1493-1503).
    n2[1] = 0.0
    n2[nz + 1] = 0.0
    if ks > 1:
        n2[ks] = max(0.0, i_dz_int[ks] * (b0[ks - 1] - b[ks]))
    for k in range(ks + 1, ke + 1):
        n2[k] = max(0.0, i_dz_int[k] * (b[k - 1] - b[k]))
    if ke < nz:
        n2[ke + 1] = max(0.0, i_dz_int[ke + 1] * (b[ke] - b0[ke + 1]))


def _find_kappa_tke(n2: np.ndarray, s2: np.ndarray, kappa_in: np.ndarray, idz: np.ndarray,
                    h_int: np.ndarray, dz_int: np.ndarray, i_l2_bdry: np.ndarray,
                    f2: float, nz: int, p: JacksonParams, kappa0: float,
                    kappa_trunc: float, k_q: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Consistent κ [m² s⁻¹] and Q [m² s⁻²] for fixed N², S² (JHL eqs. 10–11).

    Port of ``find_kappa_tke`` (MOM_kappa_shear.F90:1701-2041): Picard
    iterations of the steady TKE equation (tridiagonal, :1709-1773) and the κ
    equation κ/L_d² − ∂²κ/∂z² = K_src (tridiagonal, :1775-1823), switching to a
    Newton solve of the coupled system (:1825-2001) when close; convergence
    test :2003-2042.  ``k_q`` (= κ/Q [s], intent(inout) in MOM6) is updated in
    place.  All arrays 1-based, top-down, length nz + 2.

    Returns
    -------
    tke, kappa, kappa_src, local_src : arrays (nz + 2,)
        Q [m² s⁻²], κ [m² s⁻¹], K_src [s⁻¹] and the local source of κ used by the
        sub-step control [s⁻¹] (:2050-2066).
    """
    n1 = nz + 2
    c_n2 = p.tke_n_decay * p.tke_n_decay
    c_s2 = p.tke_shear_decay * p.tke_shear_decay
    q0 = p.tke_background
    tke_min = max(q0, TKE_FLOOR)
    ri_crit = p.ri_crit
    i_lambda2 = 1.0 / (p.kappa_buoy_scale * p.kappa_buoy_scale)
    tol_err = p.tol_err
    do_newton = False
    abort_newton = False
    newton_err = 0.2  # (:1638) may be reduced below

    ks_kappa, ke_kappa, ks_kappa_prev, ke_kappa_prev = 2, nz, 2, nz

    k_src = np.zeros(n1)
    kappa_src_out = np.zeros(n1)
    local_src = np.zeros(n1)
    ke_src, ks_src = 0, nz + 1
    for kk in range(2, nz + 1):
        if n2[kk] < ri_crit * s2[kk]:  # Ri < Ri_c
            k_src[kk] = _k_source(n2[kk], s2[kk], p)
            ke_src = kk
            if ks_src > kk:
                ks_src = kk
        else:
            k_src[kk] = 0.0

    # No source anywhere: κ = 0, Q = Q_min (:1657-1665).
    if ks_src > ke_src:
        k_q[1:nz + 2] = 0.0
        tke = np.full(n1, tke_min)
        tke[0] = 0.0
        return tke, np.zeros(n1), kappa_src_out, local_src

    # Locals that persist across iterations within this call.
    aQ = np.zeros(n1)
    dQdz = np.zeros(n1)
    dK = np.zeros(n1)
    dQ = np.zeros(n1)
    cQ = np.zeros(n1)
    cK = np.zeros(n1)
    i_ld2 = np.zeros(n1)
    dQmdK = np.zeros(n1)
    dKdQ = np.zeros(n1)
    e1 = np.zeros(n1)

    kappa = kappa_in.copy()
    tke = np.zeros(n1)
    # D = √(c_N² N² + c_S² S²) replaces JHL c_N N + c_S S (:1668-1670).
    tke_decay = np.zeros(n1)
    for kk in range(1, nz + 2):
        tke_decay[kk] = np.sqrt(c_n2 * n2[kk] + c_s2 * s2[kk])
        if kappa[kk] > 0.0 and k_q[kk] > 0.0:
            tke[kk] = kappa[kk] / k_q[kk]
        else:
            tke[kk] = tke_min
    kappa[1] = 0.0
    kappa[nz + 1] = 0.0

    # e1: fractional change of Q below the deepest nonzero κ, dQ(K+1) = e1 dQ(K)
    # (:1680-1697); no-flux bottom BC off (:1608-1609).
    eden2 = kappa0 * idz[nz]
    e1[nz + 1] = 0.0
    ome = 1.0
    for kk in range(nz, 1, -1):
        eden1 = h_int[kk] * tke_decay[kk] + ome * eden2
        eden2 = kappa0 * idz[kk - 1]
        i_eden = 1.0 / (eden2 + eden1)
        e1[kk] = eden2 * i_eden
        ome = eden1 * i_eden
    e1[1] = 0.0

    for itt in range(1, p.max_rino_it + 1):
        if not do_newton:
            # ---- TKE (Picard): explicit sources, implicit linearised sinks (:1709-1773)
            ke_tke = max(ke_kappa, ke_kappa_prev) + 1
            for kk in range(1, min(ke_tke, nz) + 1):
                aQ[kk] = (0.5 * (kappa[kk] + kappa[kk + 1]) + kappa0) * idz[kk]
            dQ[1] = -tke[1]
            tke[1] = q0          # Dirichlet Q = q0 at the surface (no-flux top BC off)
            cQ[2] = 0.0
            cQcomp = 1.0
            for kk in range(2, ke_tke):
                dQ[kk] = -tke[kk]
                tke_src = (kappa[kk] + kappa0) * s2[kk] + q0 * tke_decay[kk]
                bQd1 = h_int[kk] * (tke_decay[kk] + n2[kk] * k_q[kk]) + cQcomp * aQ[kk - 1]
                bQ = 1.0 / (bQd1 + aQ[kk])
                tke[kk] = bQ * (h_int[kk] * tke_src + aQ[kk - 1] * tke[kk - 1])
                cQ[kk + 1] = aQ[kk] * bQ
                cQcomp = bQd1 * bQ
            if ke_tke == nz + 1:
                tke[nz + 1] = tke_min  # Q = Q_min at the bed
                dQ[nz + 1] = 0.0
            else:
                kk = ke_tke
                tke_src = kappa0 * s2[kk] + q0 * tke_decay[kk]  # kappa(ke_tke) = 0
                bQ = 1.0 / ((h_int[kk] * tke_decay[kk] + cQcomp * aQ[kk - 1]) + aQ[kk])
                cQ[kk + 1] = aQ[kk] * bQ
                dQ[kk] = -tke[kk]
                tke[kk] = max((bQ * (h_int[kk] * tke_src + aQ[kk - 1] * tke[kk - 1])
                               + cQ[kk + 1] * (tke[kk + 1] - e1[kk + 1] * tke[kk]))
                              / (1.0 - cQ[kk + 1] * e1[kk + 1]), tke_min)
                dQ[kk] = tke[kk] + dQ[kk]
                # Adjust TKE deeper in the column (:1758-1766).
                kk = ke_tke + 1
                while kk <= nz + 1:
                    dQ[kk] = e1[kk] * dQ[kk - 1]
                    tke[kk] = max(tke[kk] + dQ[kk], tke_min)
                    if abs(dQ[kk]) < ROUNDOFF * tke[kk]:
                        break
                    kk += 1
                # After a completed Fortran loop the index is nz + 2.
                for k2 in range(kk + 1, nz + 1):
                    if dQ[k2] == 0.0:
                        break
                    dQ[k2] = 0.0
            for kk in range(ke_tke - 1, 0, -1):
                tke[kk] = max(tke[kk] + cQ[kk + 1] * tke[kk + 1], tke_min)
                dQ[kk] = tke[kk] + dQ[kk]

            # ---- κ (Picard): κ/L_d² − ∂²κ/∂z² = K_src (:1775-1823)
            ke_kappa_prev = ke_kappa
            ks_kappa_prev = ks_kappa
            dK[1] = 0.0
            cK[2] = 0.0
            cKcomp = 1.0
            if itt == 1:
                for kk in range(2, nz + 1):
                    i_ld2[kk] = (n2[kk] * i_lambda2 + f2) / tke[kk] + i_l2_bdry[kk]
            for kk in range(2, nz + 1):
                dK[kk] = -kappa[kk]
                if itt > 1:
                    i_ld2[kk] = (n2[kk] * i_lambda2 + f2) / tke[kk] + i_l2_bdry[kk]
                bKd1 = h_int[kk] * i_ld2[kk] + cKcomp * idz[kk - 1]
                bK = 1.0 / (bKd1 + idz[kk])
                kappa[kk] = bK * (idz[kk - 1] * kappa[kk - 1] + h_int[kk] * k_src[kk])
                cK[kk + 1] = idz[kk] * bK
                cKcomp = bKd1 * bK
                # Neglect values smaller than kappa_trunc (:1797-1806).
                if kappa[kk] < cKcomp * kappa_trunc:
                    kappa[kk] = 0.0
                    if kk > ke_src:
                        ke_kappa = kk - 1
                        k_q[kk] = 0.0
                        break
                elif kappa[kk] < 2.0 * cKcomp * kappa_trunc:
                    kappa[kk] = 2.0 * (kappa[kk] - cKcomp * kappa_trunc)
            k_q[ke_kappa] = kappa[ke_kappa] / tke[ke_kappa]
            dK[ke_kappa] = dK[ke_kappa] + kappa[ke_kappa]
            for kk in range(ke_kappa + 2, ke_kappa_prev + 1):
                dK[kk] = -kappa[kk]
                kappa[kk] = 0.0
                k_q[kk] = 0.0
            for kk in range(ke_kappa - 1, 1, -1):
                kappa[kk] = kappa[kk] + cK[kk + 1] * kappa[kk + 1]
                if kappa[kk] <= kappa_trunc:
                    kappa[kk] = 0.0
                    if kk < ks_src:
                        ks_kappa = kk + 1
                        k_q[kk] = 0.0
                        break
                elif kappa[kk] < 2.0 * kappa_trunc:
                    kappa[kk] = 2.0 * (kappa[kk] - kappa_trunc)
                dK[kk] = dK[kk] + kappa[kk]
                k_q[kk] = kappa[kk] / tke[kk]
            for kk in range(ks_kappa_prev, ks_kappa - 1):
                kappa[kk] = 0.0
                k_q[kk] = 0.0

        else:
            # ---- Newton solve of the coupled κ–Q system (:1825-2001)
            ks_kappa_prev = ks_kappa
            ke_kappa_prev = ke_kappa
            ke_kappa = nz
            ks_kappa = 2
            dK[1] = 0.0
            cK[2] = 0.0
            cKcomp = 1.0
            dKdQ[1] = 0.0
            aQ[1] = (0.5 * (kappa[1] + kappa[2]) + kappa0) * idz[1]
            dQdz[1] = 0.5 * (tke[1] - tke[2]) * idz[1]
            dQ[1] = 0.0
            cQ[2] = 0.0
            cQcomp = 1.0
            dQmdK[2] = 0.0
            for kk in range(2, nz + 1):
                i_q = 1.0 / tke[kk]
                i_ld2[kk] = (n2[kk] * i_lambda2 + f2) * i_q + i_l2_bdry[kk]
                kap_src = h_int[kk] * (k_src[kk] - i_ld2[kk] * kappa[kk]) \
                    + idz[kk - 1] * (kappa[kk - 1] - kappa[kk]) \
                    - idz[kk] * (kappa[kk] - kappa[kk + 1])
                # Positive pivot and 0 ≤ cK ≤ 1, otherwise no Newton (:1852-1856, 1887-1889).
                decay_term_k = -idz[kk - 1] * dQmdK[kk] * dKdQ[kk - 1] + h_int[kk] * i_ld2[kk]
                if decay_term_k < 0.0:
                    abort_newton = True
                    break
                bK = 1.0 / (idz[kk] + idz[kk - 1] * cKcomp + decay_term_k)
                cK[kk + 1] = bK * idz[kk]
                cKcomp = bK * (idz[kk - 1] * cKcomp + decay_term_k)
                # Corrected dκ/dQ with dz_Int (KAPPA_SHEAR_ITER_BUG = False, :1860-1866).
                dKdQ[kk] = bK * (idz[kk - 1] * dKdQ[kk - 1] * cQ[kk]
                                 + dz_int[kk] * (n2[kk] * i_lambda2 + f2) * (i_q * i_q) * kappa[kk])
                dK[kk] = bK * (kap_src + idz[kk - 1] * dK[kk - 1] + idz[kk - 1] * dKdQ[kk - 1] * dQ[kk - 1])
                if dK[kk] <= cKcomp * (kappa_trunc - kappa[kk]):
                    dK[kk] = -cKcomp * kappa[kk]
                elif dK[kk] < cKcomp * (2.0 * kappa_trunc - kappa[kk]):
                    dK[kk] = 2.0 * dK[kk] - cKcomp * (2.0 * kappa_trunc - kappa[kk])

                aQ[kk] = (0.5 * (kappa[kk] + kappa[kk + 1]) + kappa0) * idz[kk]
                dQdz[kk] = 0.5 * (tke[kk] - tke[kk + 1]) * idz[kk]
                tke_src = h_int[kk] * (((kappa[kk] + kappa0) * s2[kk] - kappa[kk] * n2[kk])
                                       - (tke[kk] - q0) * tke_decay[kk]) \
                    - (aQ[kk] * (tke[kk] - tke[kk + 1]) - aQ[kk - 1] * (tke[kk - 1] - tke[kk]))
                v1 = aQ[kk - 1] + dQdz[kk - 1] * dKdQ[kk - 1]
                v2 = (v1 * dQmdK[kk] + dQdz[kk - 1] * cK[kk]) \
                    + ((dQdz[kk - 1] - dQdz[kk]) + dz_int[kk] * (s2[kk] - n2[kk]))
                decay_term_q = h_int[kk] * tke_decay[kk] - dQdz[kk - 1] * dKdQ[kk - 1] * cQ[kk] \
                    - v2 * dKdQ[kk]
                if decay_term_q < 0.0:
                    abort_newton = True
                    break
                bQ = 1.0 / (aQ[kk] + (cQcomp * aQ[kk - 1] + decay_term_q))
                cQ[kk + 1] = aQ[kk] * bQ
                cQcomp = (cQcomp * aQ[kk - 1] + decay_term_q) * bQ
                dQmdK[kk + 1] = (v2 * cK[kk + 1] - dQdz[kk]) * bQ
                # Ensure that TKE + dQ will not drop below ½ TKE.
                dQ[kk] = max(bQ * ((v1 * dQ[kk - 1] + dQdz[kk - 1] * dK[kk - 1])
                                   + (v2 * dK[kk] + tke_src)), cQcomp * (-0.5 * tke[kk]))
                if itt > 1 and kk > ke_src and dK[kk] == 0.0 and (kappa[kk] + kappa[kk + 1]) == 0.0:
                    ke_kappa = kk - 1
                    break
            if ke_kappa == nz and not abort_newton:
                dK[nz + 1] = 0.0
                dKdQ[nz + 1] = 0.0
                dQ[nz + 1] = 0.0  # no-flux bottom BC off
            elif not abort_newton:
                dQ[ke_kappa + 1] = dQ[ke_kappa + 1] / (1.0 - cQ[ke_kappa + 2] * e1[ke_kappa + 2])
                tke[ke_kappa + 1] = max(tke[ke_kappa + 1] + dQ[ke_kappa + 1], tke_min)
                for kk in range(ke_kappa + 2, nz + 2):
                    dK[kk] = 0.0
                    dQ[kk] = max(e1[kk] * dQ[kk - 1], -0.5 * tke[kk])
                    tke[kk] = max(tke[kk] + dQ[kk], tke_min)
                    if abs(dQ[kk]) < ROUNDOFF * tke[kk]:
                        break
            if not abort_newton:
                for kk in range(ke_kappa, 1, -1):
                    dQ[kk] = max(dQ[kk] + (cQ[kk + 1] * dQ[kk + 1] + dQmdK[kk + 1] * dK[kk + 1]),
                                 -0.5 * tke[kk])
                    tke[kk] = max(tke[kk] + dQ[kk], tke_min)
                    dK[kk] = dK[kk] + (cK[kk + 1] * dK[kk + 1] + dKdQ[kk] * dQ[kk])
                    if dK[kk] <= kappa_trunc - kappa[kk]:
                        dK[kk] = -kappa[kk]
                        kappa[kk] = 0.0
                        if kk < ks_src and kk + 1 > ks_kappa:
                            ks_kappa = kk + 1
                    elif dK[kk] < 2.0 * kappa_trunc - kappa[kk]:
                        dK[kk] = 2.0 * dK[kk] - (2.0 * kappa_trunc - kappa[kk])
                        kappa[kk] = max(kappa[kk] + dK[kk], 0.0)
                        if kk <= ks_kappa:
                            ks_kappa = 2
                    else:
                        kappa[kk] = kappa[kk] + dK[kk]
                        if kk <= ks_kappa:
                            ks_kappa = 2
                dQ[1] = max(dQ[1] + cQ[2] * dQ[2] + dQmdK[2] * dK[2], tke_min - tke[1])
                tke[1] = max(tke[1] + dQ[1], tke_min)
                dK[1] = 0.0

        # ---- Convergence test (:2003-2042)
        k_lo = min(ks_kappa, ks_kappa_prev)
        k_hi = max(ke_kappa, ke_kappa_prev)
        if tol_err < newton_err and not abort_newton:
            # A lower tolerance is used to switch to Newton than to switch back.
            newton_test = 2.0 * newton_err if do_newton else newton_err
            within_tolerance = True
            do_newton = True
            for kk in range(k_lo, k_hi + 1):
                kappa_mean = kappa0 + (kappa[kk] - 0.5 * dK[kk])
                if abs(dK[kk]) > newton_test * kappa_mean:
                    if do_newton:
                        abort_newton = True
                    within_tolerance = False
                    do_newton = False
                    break
                elif abs(dK[kk]) > tol_err * kappa_mean:
                    within_tolerance = False
                    if not do_newton:
                        break
                if abs(dQ[kk]) > newton_test * (tke[kk] - 0.5 * dQ[kk]):
                    if do_newton:
                        abort_newton = True
                    do_newton = False
                    if not within_tolerance:
                        break
        else:
            within_tolerance = True
            for kk in range(k_lo, k_hi + 1):
                if abs(dK[kk]) > tol_err * (kappa0 + (kappa[kk] - 0.5 * dK[kk])):
                    within_tolerance = False
                    break

        if abort_newton:
            do_newton = False
            abort_newton = False
            newton_err = 0.5 * newton_err  # went to Newton too quickly
            ke_kappa_prev = nz
            for kk in range(2, nz + 1):
                k_q[kk] = kappa[kk] / max(tke[kk], tke_min)

        if within_tolerance:
            break

    if do_newton:  # K_Q needs to be calculated (:2044-2048)
        k_q[1:ks_kappa] = 0.0
        for kk in range(ks_kappa, ke_kappa + 1):
            k_q[kk] = kappa[kk] / tke[kk]
        k_q[ke_kappa + 1:nz + 2] = 0.0

    # Local source of κ for the sub-step control (:2050-2060).
    for kk in range(2, nz + 1):
        diffusive_src = idz[kk - 1] * (kappa[kk - 1] - kappa[kk]) + idz[kk] * (kappa[kk + 1] - kappa[kk])
        chg_by_k0 = kappa0 * ((idz[kk - 1] + idz[kk]) / h_int[kk] + i_ld2[kk])
        if diffusive_src <= 0.0:
            local_src[kk] = k_src[kk] + chg_by_k0
        else:
            local_src[kk] = (k_src[kk] + chg_by_k0) + diffusive_src / h_int[kk]
    kappa_src_out[2:nz + 1] = k_src[2:nz + 1]
    return tke, kappa, kappa_src_out, local_src


def kappa_shear_column(u: np.ndarray, v: np.ndarray, b: np.ndarray, dz: np.ndarray,
                       f2: float, dt: float, kappa0: float,
                       p: JacksonParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Δt-averaged shear-driven diffusivity κ and TKE Q of one column
    (Jackson et al. 2008; port of MOM6 ``kappa_shear_column``,
    MOM_kappa_shear.F90:866-1374).

    Parameters
    ----------
    u, v : (N,) velocity at the layer centres [m s⁻¹], k = 0 at the bed.
    b : (N,) buoyancy −g(ρ − ρ₀)/ρ₀ of each layer [m s⁻²] (one tracer in place
        of MOM6's T, S; see the module docstring).
    dz : (N,) layer thicknesses [m], all > 0 (no thin-layer merging here).
    f2 : f² of the column [s⁻²].
    dt : time step over which κ is averaged [s].
    kappa0 : background diffusivity κ0 = KD_KAPPA_SHEAR_0 [m² s⁻¹], > 0; the
        caller passes max(kv, 1e-7) as MOM6 (:2170-2178).  It smooths u, v, b
        before the gradients, enters the TKE equation and sets
        κ_trunc = ``kd_trunc_frac`` κ0; it is **not** included in the result.
    p : JacksonParams.

    Returns
    -------
    kappa, tke : (N + 1,) arrays at the interfaces, k = 0 at the bed
        κ averaged over dt [m² s⁻¹] (0 at the bed and the surface) and
        Q averaged over dt [m² s⁻²].
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    b = np.asarray(b, dtype=float)
    dz = np.asarray(dz, dtype=float)
    if dz.ndim != 1 or dz.size < 2:
        raise ValueError("kappa_shear_column: dz must be (N,) with N >= 2")
    nz = dz.size
    for name, arr in (("u", u), ("v", v), ("b", b)):
        if arr.shape != dz.shape:
            raise ValueError(f"kappa_shear_column: {name} must have shape {dz.shape}, got {arr.shape}")
    if not np.all(dz > 0.0):
        raise ValueError("kappa_shear_column: every layer must be thicker than 0 (merge thin layers first)")
    if not kappa0 > 0.0:
        raise ValueError("kappa_shear_column: kappa0 must be > 0 (MOM6 uses max(KD, 1e-7))")
    if not dt > 0.0:
        raise ValueError("kappa_shear_column: dt must be > 0")

    n1 = nz + 2

    def top_down(x: np.ndarray) -> np.ndarray:
        out = np.zeros(n1)
        out[1:nz + 1] = x[::-1]
        return out

    hlay = top_down(dz)                      # h = dz in the Boussinesq form [m]
    u0xdz = top_down(u) * hlay
    v0xdz = top_down(v) * hlay
    b0xdz = top_down(b) * hlay

    ri_crit = p.ri_crit
    k0dt = dt * kappa0                       # [m²]
    kappa_trunc = p.kd_trunc_frac * kappa0   # KD_TRUNC_KAPPA_SHEAR [m² s⁻¹]
    i_lz_rescale_sqr = 1.0 / (p.lz_rescale * p.lz_rescale) if p.lz_rescale > 0 else 1.0
    tol_dksrc = p.max_kap_src_chg
    # (:1014-1019) the default 10 gives exactly 0.95 without round-off.
    tol_dksrc_low = 0.95 if tol_dksrc == 10.0 else (tol_dksrc - 0.5) / tol_dksrc
    tol2 = 2.0 * p.tol_err

    # Grid metrics (:1026-1049).
    idz = np.zeros(n1)
    idz[1:nz + 1] = 1.0 / hlay[1:nz + 1]
    i_dz_int = np.zeros(n1)                  # 1 / distance between layer centres [m⁻¹]
    dist_from_top = np.zeros(n1)
    i_dz_int[1] = 2.0 / hlay[1]
    for kk in range(2, nz + 1):
        i_dz_int[kk] = 2.0 / (hlay[kk - 1] + hlay[kk])
        dist_from_top[kk] = dist_from_top[kk - 1] + hlay[kk - 1]
    i_dz_int[nz + 1] = 2.0 / hlay[nz]
    # 1/L_bdry² = [(d_top + d_bot)/(d_top d_bot)]² / LZ_RESCALE²  (Boussinesq, :1039-1049)
    i_l2_bdry = np.zeros(n1)
    dist_from_bot = 0.0
    for kk in range(nz, 1, -1):
        dist_from_bot += hlay[kk]
        dt_ = dist_from_top[kk]
        i_l2_bdry[kk] = ((dt_ + dist_from_bot) * (dt_ + dist_from_bot)) / \
            ((dt_ * dist_from_bot) * (dt_ * dist_from_bot))
        i_l2_bdry[kk] = i_lz_rescale_sqr * i_l2_bdry[kk]

    # One step of background κ0 diffusion of u, v, b (:1052-1088): no-slip
    # u, v at the bed, insulating b.
    uu = np.zeros(n1)
    vv = np.zeros(n1)
    bb = np.zeros(n1)
    a1 = np.zeros(n1)
    c1 = np.zeros(n1)
    a1[2] = k0dt * i_dz_int[2]
    b1 = 1.0 / (hlay[1] + a1[2])
    uu[1] = b1 * u0xdz[1]
    vv[1] = b1 * v0xdz[1]
    bb[1] = b1 * b0xdz[1]
    c1[2] = a1[2] * b1
    d1 = hlay[1] * b1
    for kk in range(2, nz):
        bd1 = hlay[kk] + d1 * a1[kk]
        a1[kk + 1] = k0dt * i_dz_int[kk + 1]
        b1 = 1.0 / (bd1 + a1[kk + 1])
        uu[kk] = b1 * (u0xdz[kk] + a1[kk] * uu[kk - 1])
        vv[kk] = b1 * (v0xdz[kk] + a1[kk] * vv[kk - 1])
        bb[kk] = b1 * (b0xdz[kk] + a1[kk] * bb[kk - 1])
        c1[kk + 1] = a1[kk + 1] * b1
        d1 = bd1 * b1
    b1 = 1.0 / ((hlay[nz] + d1 * a1[nz]) + k0dt * i_dz_int[nz + 1])
    uu[nz] = b1 * (u0xdz[nz] + a1[nz] * uu[nz - 1])
    vv[nz] = b1 * (v0xdz[nz] + a1[nz] * vv[nz - 1])
    b1 = 1.0 / (hlay[nz] + d1 * a1[nz])
    bb[nz] = b1 * (b0xdz[nz] + a1[nz] * bb[nz - 1])
    for kk in range(nz - 1, 0, -1):
        uu[kk] = uu[kk] + c1[kk + 1] * uu[kk + 1]
        vv[kk] = vv[kk] + c1[kk + 1] * vv[kk + 1]
        bb[kk] = bb[kk] + c1[kk + 1] * bb[kk + 1]

    # Interface cells: harmonic-mean split of each layer (:1091-1109).  Thin
    # layers give thin interface cells; Σ h_Int = column thickness; the top and
    # bottom interfaces have zero size.  Boussinesq: dz_Int = h_Int.
    h_int = np.zeros(n1)
    h_int[2] = hlay[1]
    for kk in range(2, nz):
        norm = 1.0 / (hlay[kk] * (hlay[kk - 1] + hlay[kk + 1]) + 2.0 * hlay[kk - 1] * hlay[kk + 1])
        wt_a = ((hlay[kk] + hlay[kk + 1]) * hlay[kk - 1]) * norm
        wt_b = ((hlay[kk - 1] + hlay[kk]) * hlay[kk + 1]) * norm
        h_int[kk] = h_int[kk] + hlay[kk] * wt_a
        h_int[kk + 1] = hlay[kk] * wt_b
    h_int[nz] = h_int[nz] + hlay[nz]
    h_int[nz + 1] = 0.0
    dz_int = h_int

    # Initial N², S² (:1167-1172, calculate_projected_state with dt = 0).
    n2 = np.zeros(n1)
    s2 = np.zeros(n1)
    kappa = np.zeros(n1)
    kappa[1:nz + 2] = p.kd_seed               # seed κ = KD_SEED_KAPPA_SHEAR (:325)
    _projected_state(kappa, uu, vv, bb, 0.0, nz, hlay, i_dz_int, uu, vv, bb, n2, s2)

    # ---- Sub-steps over dt (:1179-1374)
    dt_rem = dt
    k_q = np.zeros(n1)
    kappa_avg = np.zeros(n1)
    tke_avg = np.zeros(n1)
    local_src_avg = np.zeros(n1)
    for kk in range(1, nz + 2):
        if h_int[kk] > 0.0:
            local_src_avg[kk] = 0.1 * k0dt * i_dz_int[kk] / h_int[kk]
    u_test = np.zeros(n1)
    v_test = np.zeros(n1)
    b_test = np.zeros(n1)
    kappa_mid = np.zeros(n1)
    tol_max = np.zeros(n1)
    tol_min = np.zeros(n1)
    tol_chg = np.zeros(n1)

    def valid_step(ks_kappa: int, ke_kappa: int, kappa_src: np.ndarray, idtt: float) -> bool:
        """K_src after the projected mixing stays within tolerance (:1240-1263)."""
        for kk in range(max(ks_kappa - 1, 2), min(ke_kappa + 1, nz) + 1):
            if n2[kk] < ri_crit * s2[kk]:
                ksrc = _k_source(n2[kk], s2[kk], p)
                # USE_RESTRICTIVE_TOLERANCE_CHECK = False (:1251-1255)
                if (ksrc > max(tol_max[kk], kappa_src[kk] + idtt * tol_chg[kk])) or \
                        (ksrc < min(tol_min[kk], kappa_src[kk] - idtt * tol_chg[kk])):
                    return False
            else:
                if 0.0 < min(tol_min[kk], kappa_src[kk] - idtt * tol_chg[kk]):
                    return False
        return True

    for itt in range(1, p.max_ks_it + 1):
        tke, kappa_out, kappa_src, local_src = _find_kappa_tke(
            n2, s2, kappa, idz, h_int, dz_int, i_l2_bdry, f2, nz, p, kappa0, kappa_trunc, k_q)

        # Range of nonzero κ_out (:1203-1212).
        ks_kappa, ke_kappa = nz + 1, 0
        for kk in range(2, nz + 1):
            if kappa_out[kk] > 0.0:
                ks_kappa = kk
                break
        for kk in range(nz, ks_kappa - 1, -1):
            if kappa_out[kk] > 0.0:
                ke_kappa = kk
                break
        if ke_kappa == nz:
            kappa_out[nz + 1] = 0.0

        # Length of this sub-step, dt_now (:1215-1302).
        if ke_kappa < ks_kappa or itt == p.max_ks_it:
            dt_now = dt_rem
        else:
            dt_test = dt_rem
            for kk in range(2, nz + 1):
                tol_max[kk] = kappa_src[kk] + tol_dksrc * local_src[kk]
                tol_min[kk] = kappa_src[kk] - tol_dksrc_low * local_src[kk]
                tol_chg[kk] = tol2 * local_src_avg[kk]
            valid_dt = False
            # Halving, fewer times as the sub-steps run out (:1232-1266).
            for _ in range((p.max_ks_it + 1 - itt) // 2):
                _projected_state(kappa_out, uu, vv, bb, 0.5 * dt_test, nz, hlay, i_dz_int,
                                 u_test, v_test, b_test, n2, s2, ks_kappa, ke_kappa)
                valid_dt = valid_step(ks_kappa, ke_kappa, kappa_src, 1.0 / dt_test)
                if valid_dt:
                    break
                dt_test = 0.5 * dt_test
            # Bisection refinements (:1267-1294).
            if dt_test < dt_rem and valid_dt:
                dt_inc = 0.5 * dt_test
                for _ in range(DT_REFINEMENTS):
                    _projected_state(kappa_out, uu, vv, bb, 0.5 * (dt_test + dt_inc), nz, hlay,
                                     i_dz_int, u_test, v_test, b_test, n2, s2, ks_kappa, ke_kappa)
                    valid_dt = valid_step(ks_kappa, ke_kappa, kappa_src, 1.0 / (dt_test + dt_inc))
                    if valid_dt:
                        dt_test = dt_test + dt_inc
                    dt_inc = 0.5 * dt_inc
            else:
                dt_inc = 0.0
            dt_now = min(dt_test * (1.0 + p.tol_err) + dt_inc, dt_rem)
            local_src_avg[2:nz + 1] += dt_now * local_src[2:nz + 1]

        if ke_kappa < ks_kappa:
            # No mixing now, and none later (:1305-1314).
            dt_wt = dt_rem / dt
            dt_rem = 0.0
            kappa_mid[:] = 0.0
            tke_avg[1:nz + 2] += dt_wt * tke[1:nz + 2]
        else:
            # Predictor (:1316-1336).
            _projected_state(kappa_out, uu, vv, bb, dt_now, nz, hlay, i_dz_int,
                             u_test, v_test, b_test, n2, s2, ks_kappa, ke_kappa)
            k_q_tmp = k_q.copy()
            tke_pred, kappa_pred, _, _ = _find_kappa_tke(
                n2, s2, kappa_out, idz, h_int, dz_int, i_l2_bdry, f2, nz, p, kappa0, kappa_trunc, k_q_tmp)
            ks_kappa, ke_kappa = nz + 1, 0
            for kk in range(1, nz + 2):
                kappa_mid[kk] = 0.5 * (kappa_out[kk] + kappa_pred[kk])
                if kappa_mid[kk] > 0.0 and kk < ks_kappa:
                    ks_kappa = kk
                if kappa_mid[kk] > 0.0:
                    ke_kappa = kk
            # Corrector (:1338-1346).
            _projected_state(kappa_mid, uu, vv, bb, dt_now, nz, hlay, i_dz_int,
                             u_test, v_test, b_test, n2, s2, ks_kappa, ke_kappa)
            tke_pred, kappa_pred, _, _ = _find_kappa_tke(
                n2, s2, kappa_out, idz, h_int, dz_int, i_l2_bdry, f2, nz, p, kappa0, kappa_trunc, k_q)
            # Time average (:1348-1357).
            dt_wt = dt_now / dt
            dt_rem = dt_rem - dt_now
            kappa_mid[1:nz + 2] = 0.5 * (kappa_out[1:nz + 2] + kappa_pred[1:nz + 2])
            kappa_avg[1:nz + 2] += kappa_mid[1:nz + 2] * dt_wt
            tke_avg[1:nz + 2] += dt_wt * 0.5 * (tke_pred[1:nz + 2] + tke[1:nz + 2])
            kappa = kappa_pred.copy()  # first guess for the next sub-step

        if dt_rem > 0.0:
            # Mix the solver's column state with κ_mid over dt_now (:1361-1368).
            _projected_state(kappa_mid, uu, vv, bb, dt_now, nz, hlay, i_dz_int,
                             uu, vv, bb, n2, s2)
        if dt_rem <= 0.0:
            break

    # Back to the model orientation: interface 0 = bed, N = surface.
    return kappa_avg[1:nz + 2][::-1].copy(), tke_avg[1:nz + 2][::-1].copy()


# =====================================================================================================================
# Columns with thin-layer merging: NumPy oracle and Numba kernel (Jackson task 3)
# =====================================================================================================================
#
# Thin layers are merged before the column solve, as MOM6 ``KAPPA_SHEAR_ELIM_MASSLESS`` (MOM_kappa_shear.F90:254-303)
# with pyOCEAN's relative threshold (``core.components.closure.thin_layer_threshold``, N3 task 9c) and clustering
# (``active_layers``): a layer with h ≥ h_thin opens a merged layer, every thinner layer joins the nearest open layer
# above it (thin layers at the surface holding ≥ h_massless together form their own); Σh, Σu·h, Σv·h, Σb·h are summed
# and u, v, b of a merged layer are their thickness-weighted means.  κ and Q go back to the model interfaces by linear
# interpolation in the fractional height inside the merged layer (MOM6 kc, kf, :286-298 and :355-363): an interface
# between merged layers takes the merged value exactly.  A dry column (Σh ≤ h_massless) or one that merges to fewer
# than two layers has no interior interface: κ = Q = 0.
#
# The Numba kernel is a line-by-line copy of ``kappa_shear_column`` / ``_find_kappa_tke`` / ``_projected_state``
# with the same operation order, so it is bit for bit with ``kappa_shear_columns_numpy``
# (tests/test_kappa_shear.py::test_numba_matches_numpy_oracle).  JacksonParams enters as a float64 array in the
# field order of the dataclass (``PARAM_FIELDS``; the two integer fields are converted back with int()).

H_MASSLESS: float = 1.0e-3  # [m] least thickness of a column / of a surface thin cluster (= closure.H_MASSLESS)

PARAM_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(JacksonParams))
(_RI_CRIT, _SHEARMIX_RATE, _FRI_CURVATURE, _KAPPA_BUOY_SCALE, _TKE_N_DECAY, _TKE_SHEAR_DECAY, _KD_SEED,
 _KD_TRUNC_FRAC, _TOL_ERR, _MAX_RINO_IT, _MAX_KS_IT, _MAX_KAP_SRC_CHG, _TKE_BACKGROUND,
 _LZ_RESCALE) = range(len(PARAM_FIELDS))


def params_array(p: JacksonParams) -> np.ndarray:
    """JacksonParams as the float64 array the Numba kernel takes, in the order ``PARAM_FIELDS``."""
    return np.array([float(getattr(p, name)) for name in PARAM_FIELDS])


def _merge_map(dzc: np.ndarray, act: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Merged layer of each model layer and the interpolation weights of one column (shared by the NumPy oracle;
    the Numba kernel repeats it in ``_merge_column_nb``).

    dzc : (N,) layer thicknesses [m]; act : (N,) layers that open a merged layer (``active_layers``).
    Returns lay (N,) merged layer of each model layer, kc (N+1,) merged layer of each model interface and kf (N+1,)
    its fractional height in that merged layer (the weight of the merged interface above, MOM6 kf), and c, the
    number of merged layers.  Needs c ≥ 1.
    """
    n = dzc.shape[0]
    lay = np.empty(n, np.int64)
    nxt = -1
    for k in range(n - 1, -1, -1):                 # owner: nearest open layer above (k' ≥ k) ...
        if act[k]:
            nxt = k
        lay[k] = nxt
    prv = -1
    rank = np.empty(n, np.int64)
    r = -1
    for k in range(n):
        if act[k]:
            prv = k
            r += 1
        rank[k] = r
        if lay[k] < 0:                             # ... else the nearest below (thin top stack < h_massless)
            lay[k] = prv
    for k in range(n):
        lay[k] = rank[lay[k]]
    c = r + 1
    hc = np.zeros(c)
    for k in range(n):
        hc[lay[k]] += dzc[k]
    kc = np.empty(n + 1, np.int64)
    kf = np.empty(n + 1)
    d_in = 0.0                                     # thickness of the merged layer below interface i [m]
    for i in range(n):
        kc[i] = lay[i]
        if i == 0 or lay[i] != lay[i - 1]:
            kf[i] = 0.0
            d_in = dzc[i]
        else:
            kf[i] = d_in / hc[lay[i]]
            d_in = d_in + dzc[i]
    kc[n] = lay[n - 1]
    kf[n] = 1.0
    return lay, kc, kf, c


def kappa_shear_columns_numpy(u: np.ndarray, v: np.ndarray, b: np.ndarray, dz: np.ndarray, f2: np.ndarray,
                              dt: float, kappa0: float, h_thin: np.ndarray, p: JacksonParams,
                              h_massless: float = H_MASSLESS, T: np.ndarray | None = None,
                              S: np.ndarray | None = None, eos=None, rho0: float = 1000.0,
                              g: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
    """
    Jackson et al. (2008) κ and Q of M columns with thin-layer merging — the **NumPy oracle** of
    ``kappa_shear_columns`` (thin-layer merge of the section header, then ``kappa_shear_column`` per column).

    Parameters
    ----------
    u, v : (N, M) layer velocities [m s⁻¹]; b : (N, M) buoyancy −g(ρ − ρ₀)/ρ₀ [m s⁻²]; dz : (N, M) layer
        thicknesses [m] ≥ 0; layers k = 0 (bed) … N−1 (surface).
    f2 : (M,) f² [s⁻²]; dt : averaging step [s]; kappa0 : κ0 [m² s⁻¹] (``kappa_shear_column``).
    h_thin : (M,) thin-layer threshold [m] (``core.components.closure.thin_layer_threshold``).
    p : JacksonParams; h_massless : [m] dry-column limit and least total of a surface thin cluster.
    T, S, eos, rho0, g : nonlinear EOS (module docstring): T, S (N, M) in the EOS convention, ``eos`` a nonlinear
        EOS component, ρ₀ [kg m⁻³] and g [m s⁻²] of the interface pressure and of g/ρ₀; ``b`` is then not used.
        ``eos`` None (linear EOS): b as before.

    Returns
    -------
    kappa, tke : (N+1, M) Δt-averaged κ [m² s⁻¹] and Q [m² s⁻²] at the model interfaces (0 = bed).
    """
    from core.components.closure import active_layers     # lazy: closure imports this module (task 4)

    code = 0 if eos is None else eos_code(eos)

    u, v, b, dz = (np.asarray(x, float) for x in (u, v, b, dz))
    f2 = np.asarray(f2, float)
    h_thin = np.asarray(h_thin, float)
    n, m = dz.shape
    kappa = np.zeros((n + 1, m))
    tke = np.zeros((n + 1, m))
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
        hu = np.zeros(c)
        hv = np.zeros(c)
        hb = np.zeros(c)
        ht = np.zeros(c)
        hs = np.zeros(c)
        for k in range(n):
            h = dzc[k]
            hc[lay[k]] += h
            hu[lay[k]] += h * u[k, j]
            hv[lay[k]] += h * v[k, j]
            if code == 0:
                hb[lay[k]] += h * b[k, j]
            else:
                ht[lay[k]] += h * T[k, j]
                hs[lay[k]] += h * S[k, j]
        for q in range(c):
            hu[q] = hu[q] / hc[q]
            hv[q] = hv[q] / hc[q]
            if code == 0:
                hb[q] = hb[q] / hc[q]
            else:
                ht[q] = ht[q] / hc[q]
                hs[q] = hs[q] / hc[q]
        if code != 0:
            _local_buoyancy(ht, hs, hc, c, code, float(rho0), float(g), hb)
        kap_c, tke_c = kappa_shear_column(hu, hv, hb, hc, float(f2[j]), dt, kappa0, p)
        for i in range(n + 1):
            lo_k, hi_k = kap_c[kc[i]], kap_c[min(kc[i] + 1, c)]
            lo_q, hi_q = tke_c[kc[i]], tke_c[min(kc[i] + 1, c)]
            f = kf[i]
            if f == 0.0:
                kappa[i, j], tke[i, j] = lo_k, lo_q
            elif f == 1.0:
                kappa[i, j], tke[i, j] = hi_k, hi_q
            else:
                kappa[i, j] = (1.0 - f) * lo_k + f * hi_k
                tke[i, j] = (1.0 - f) * lo_q + f * hi_q
    return kappa, tke


# ------------------------------------------------------------------------------------------- Numba port of the oracle
@njit(cache=True)
def _k_source_nb(n2: float, s2: float, prm: np.ndarray) -> float:
    """``_k_source`` (JHL eq. 10 source, MOM_kappa_shear.F90:1646-1652) [s⁻¹]."""
    rc = prm[_RI_CRIT]
    return (2.0 * prm[_SHEARMIX_RATE] * np.sqrt(s2)) * \
        ((rc * s2 - n2) / (rc * s2 + prm[_FRI_CURVATURE] * n2))


@njit(cache=True)
def _projected_state_nb(kappa, u0in, v0in, b0in, dt, nz, dz, i_dz_int, u, v, b, n2, s2, ks_int, ke_int):
    """``_projected_state`` (MOM_kappa_shear.F90:1379-1506); ks_int, ke_int = −1 stand for None."""
    ks, ke = 1, nz
    if ks_int >= 0:
        ks = max(ks_int - 1, 1)
    if ke_int >= 0:
        ke = min(ke_int, nz)
    if ks > ke:
        return
    u0 = u0in.copy()
    v0 = v0in.copy()
    b0 = b0in.copy()

    if dt > 0.0:
        c1 = np.zeros(nz + 2)
        a_b = dt * (kappa[ks + 1] * i_dz_int[ks + 1])
        b1 = 1.0 / (dz[ks] + a_b)
        c1[ks + 1] = a_b * b1
        d1 = dz[ks] * b1
        u[ks] = (b1 * dz[ks]) * u0[ks]
        v[ks] = (b1 * dz[ks]) * v0[ks]
        b[ks] = (b1 * dz[ks]) * b0[ks]
        for k in range(ks + 1, ke):
            a_a = a_b
            a_b = dt * (kappa[k + 1] * i_dz_int[k + 1])
            bd1 = dz[k] + d1 * a_a
            b1 = 1.0 / (bd1 + a_b)
            c1[k + 1] = a_b * b1
            d1 = bd1 * b1
            u[k] = b1 * (dz[k] * u0[k] + a_a * u[k - 1])
            v[k] = b1 * (dz[k] * v0[k] + a_a * v[k - 1])
            b[k] = b1 * (dz[k] * b0[k] + a_a * b[k - 1])
        a_a = a_b
        b1 = 1.0 / (dz[ke] + d1 * a_a)
        b[ke] = b1 * (dz[ke] * b0[ke] + a_a * b[ke - 1])
        if ke == nz:
            a_b = dt * (kappa[nz + 1] * i_dz_int[nz + 1])
            b1nz_0 = 1.0 / ((dz[nz] + d1 * a_a) + a_b)
        else:
            b1nz_0 = b1
        u[ke] = b1nz_0 * (dz[ke] * u0[ke] + a_a * u[ke - 1])
        v[ke] = b1nz_0 * (dz[ke] * v0[ke] + a_a * v[ke - 1])
        for k in range(ke - 1, ks - 1, -1):
            u[k] = u[k] + c1[k + 1] * u[k + 1]
            v[k] = v[k] + c1[k + 1] * v[k + 1]
            b[k] = b[k] + c1[k + 1] * b[k + 1]
    else:
        for k in range(1, nz + 1):
            u[k] = u0[k]
            v[k] = v0[k]
            b[k] = b0[k]

    s2[1] = 0.0
    s2[nz + 1] = 0.0
    if ks > 1:
        s2[ks] = ((u[ks] - u0[ks - 1]) * (u[ks] - u0[ks - 1]) + (v[ks] - v0[ks - 1]) * (v[ks] - v0[ks - 1])) \
            * (i_dz_int[ks] * i_dz_int[ks])
    for k in range(ks + 1, ke + 1):
        s2[k] = ((u[k] - u[k - 1]) * (u[k] - u[k - 1]) + (v[k] - v[k - 1]) * (v[k] - v[k - 1])) \
            * (i_dz_int[k] * i_dz_int[k])
    if ke < nz:
        s2[ke + 1] = ((u0[ke + 1] - u[ke]) * (u0[ke + 1] - u[ke]) + (v0[ke + 1] - v[ke]) * (v0[ke + 1] - v[ke])) \
            * (i_dz_int[ke + 1] * i_dz_int[ke + 1])

    n2[1] = 0.0
    n2[nz + 1] = 0.0
    if ks > 1:
        n2[ks] = max(0.0, i_dz_int[ks] * (b0[ks - 1] - b[ks]))
    for k in range(ks + 1, ke + 1):
        n2[k] = max(0.0, i_dz_int[k] * (b[k - 1] - b[k]))
    if ke < nz:
        n2[ke + 1] = max(0.0, i_dz_int[ke + 1] * (b[ke] - b0[ke + 1]))


@njit(cache=True)
def _find_kappa_tke_nb(n2, s2, kappa_in, idz, h_int, dz_int, i_l2_bdry, f2, nz, prm, kappa0, kappa_trunc, k_q):
    """``_find_kappa_tke`` (MOM_kappa_shear.F90:1701-2041), same operation order; returns (Q, κ, K_src, local)."""
    n1 = nz + 2
    c_n2 = prm[_TKE_N_DECAY] * prm[_TKE_N_DECAY]
    c_s2 = prm[_TKE_SHEAR_DECAY] * prm[_TKE_SHEAR_DECAY]
    q0 = prm[_TKE_BACKGROUND]
    tke_min = max(q0, TKE_FLOOR)
    ri_crit = prm[_RI_CRIT]
    i_lambda2 = 1.0 / (prm[_KAPPA_BUOY_SCALE] * prm[_KAPPA_BUOY_SCALE])
    tol_err = prm[_TOL_ERR]
    max_rino_it = int(prm[_MAX_RINO_IT])
    do_newton = False
    abort_newton = False
    newton_err = 0.2

    ks_kappa, ke_kappa, ks_kappa_prev, ke_kappa_prev = 2, nz, 2, nz

    k_src = np.zeros(n1)
    kappa_src_out = np.zeros(n1)
    local_src = np.zeros(n1)
    ke_src, ks_src = 0, nz + 1
    for kk in range(2, nz + 1):
        if n2[kk] < ri_crit * s2[kk]:
            k_src[kk] = _k_source_nb(n2[kk], s2[kk], prm)
            ke_src = kk
            if ks_src > kk:
                ks_src = kk
        else:
            k_src[kk] = 0.0

    if ks_src > ke_src:
        for kk in range(1, nz + 2):
            k_q[kk] = 0.0
        tke0 = np.full(n1, tke_min)
        tke0[0] = 0.0
        return tke0, np.zeros(n1), kappa_src_out, local_src

    aQ = np.zeros(n1)
    dQdz = np.zeros(n1)
    dK = np.zeros(n1)
    dQ = np.zeros(n1)
    cQ = np.zeros(n1)
    cK = np.zeros(n1)
    i_ld2 = np.zeros(n1)
    dQmdK = np.zeros(n1)
    dKdQ = np.zeros(n1)
    e1 = np.zeros(n1)

    kappa = kappa_in.copy()
    tke = np.zeros(n1)
    tke_decay = np.zeros(n1)
    for kk in range(1, nz + 2):
        tke_decay[kk] = np.sqrt(c_n2 * n2[kk] + c_s2 * s2[kk])
        if kappa[kk] > 0.0 and k_q[kk] > 0.0:
            tke[kk] = kappa[kk] / k_q[kk]
        else:
            tke[kk] = tke_min
    kappa[1] = 0.0
    kappa[nz + 1] = 0.0

    eden2 = kappa0 * idz[nz]
    e1[nz + 1] = 0.0
    ome = 1.0
    for kk in range(nz, 1, -1):
        eden1 = h_int[kk] * tke_decay[kk] + ome * eden2
        eden2 = kappa0 * idz[kk - 1]
        i_eden = 1.0 / (eden2 + eden1)
        e1[kk] = eden2 * i_eden
        ome = eden1 * i_eden
    e1[1] = 0.0

    within_tolerance = False
    for itt in range(1, max_rino_it + 1):
        if not do_newton:
            ke_tke = max(ke_kappa, ke_kappa_prev) + 1
            for kk in range(1, min(ke_tke, nz) + 1):
                aQ[kk] = (0.5 * (kappa[kk] + kappa[kk + 1]) + kappa0) * idz[kk]
            dQ[1] = -tke[1]
            tke[1] = q0
            cQ[2] = 0.0
            cQcomp = 1.0
            for kk in range(2, ke_tke):
                dQ[kk] = -tke[kk]
                tke_src = (kappa[kk] + kappa0) * s2[kk] + q0 * tke_decay[kk]
                bQd1 = h_int[kk] * (tke_decay[kk] + n2[kk] * k_q[kk]) + cQcomp * aQ[kk - 1]
                bQ = 1.0 / (bQd1 + aQ[kk])
                tke[kk] = bQ * (h_int[kk] * tke_src + aQ[kk - 1] * tke[kk - 1])
                cQ[kk + 1] = aQ[kk] * bQ
                cQcomp = bQd1 * bQ
            if ke_tke == nz + 1:
                tke[nz + 1] = tke_min
                dQ[nz + 1] = 0.0
            else:
                kk = ke_tke
                tke_src = kappa0 * s2[kk] + q0 * tke_decay[kk]
                bQ = 1.0 / ((h_int[kk] * tke_decay[kk] + cQcomp * aQ[kk - 1]) + aQ[kk])
                cQ[kk + 1] = aQ[kk] * bQ
                dQ[kk] = -tke[kk]
                tke[kk] = max((bQ * (h_int[kk] * tke_src + aQ[kk - 1] * tke[kk - 1])
                               + cQ[kk + 1] * (tke[kk + 1] - e1[kk + 1] * tke[kk]))
                              / (1.0 - cQ[kk + 1] * e1[kk + 1]), tke_min)
                dQ[kk] = tke[kk] + dQ[kk]
                kk = ke_tke + 1
                while kk <= nz + 1:
                    dQ[kk] = e1[kk] * dQ[kk - 1]
                    tke[kk] = max(tke[kk] + dQ[kk], tke_min)
                    if abs(dQ[kk]) < ROUNDOFF * tke[kk]:
                        break
                    kk += 1
                for k2 in range(kk + 1, nz + 1):
                    if dQ[k2] == 0.0:
                        break
                    dQ[k2] = 0.0
            for kk in range(ke_tke - 1, 0, -1):
                tke[kk] = max(tke[kk] + cQ[kk + 1] * tke[kk + 1], tke_min)
                dQ[kk] = tke[kk] + dQ[kk]

            ke_kappa_prev = ke_kappa
            ks_kappa_prev = ks_kappa
            dK[1] = 0.0
            cK[2] = 0.0
            cKcomp = 1.0
            if itt == 1:
                for kk in range(2, nz + 1):
                    i_ld2[kk] = (n2[kk] * i_lambda2 + f2) / tke[kk] + i_l2_bdry[kk]
            for kk in range(2, nz + 1):
                dK[kk] = -kappa[kk]
                if itt > 1:
                    i_ld2[kk] = (n2[kk] * i_lambda2 + f2) / tke[kk] + i_l2_bdry[kk]
                bKd1 = h_int[kk] * i_ld2[kk] + cKcomp * idz[kk - 1]
                bK = 1.0 / (bKd1 + idz[kk])
                kappa[kk] = bK * (idz[kk - 1] * kappa[kk - 1] + h_int[kk] * k_src[kk])
                cK[kk + 1] = idz[kk] * bK
                cKcomp = bKd1 * bK
                if kappa[kk] < cKcomp * kappa_trunc:
                    kappa[kk] = 0.0
                    if kk > ke_src:
                        ke_kappa = kk - 1
                        k_q[kk] = 0.0
                        break
                elif kappa[kk] < 2.0 * cKcomp * kappa_trunc:
                    kappa[kk] = 2.0 * (kappa[kk] - cKcomp * kappa_trunc)
            k_q[ke_kappa] = kappa[ke_kappa] / tke[ke_kappa]
            dK[ke_kappa] = dK[ke_kappa] + kappa[ke_kappa]
            for kk in range(ke_kappa + 2, ke_kappa_prev + 1):
                dK[kk] = -kappa[kk]
                kappa[kk] = 0.0
                k_q[kk] = 0.0
            for kk in range(ke_kappa - 1, 1, -1):
                kappa[kk] = kappa[kk] + cK[kk + 1] * kappa[kk + 1]
                if kappa[kk] <= kappa_trunc:
                    kappa[kk] = 0.0
                    if kk < ks_src:
                        ks_kappa = kk + 1
                        k_q[kk] = 0.0
                        break
                elif kappa[kk] < 2.0 * kappa_trunc:
                    kappa[kk] = 2.0 * (kappa[kk] - kappa_trunc)
                dK[kk] = dK[kk] + kappa[kk]
                k_q[kk] = kappa[kk] / tke[kk]
            for kk in range(ks_kappa_prev, ks_kappa - 1):
                kappa[kk] = 0.0
                k_q[kk] = 0.0

        else:
            ks_kappa_prev = ks_kappa
            ke_kappa_prev = ke_kappa
            ke_kappa = nz
            ks_kappa = 2
            dK[1] = 0.0
            cK[2] = 0.0
            cKcomp = 1.0
            dKdQ[1] = 0.0
            aQ[1] = (0.5 * (kappa[1] + kappa[2]) + kappa0) * idz[1]
            dQdz[1] = 0.5 * (tke[1] - tke[2]) * idz[1]
            dQ[1] = 0.0
            cQ[2] = 0.0
            cQcomp = 1.0
            dQmdK[2] = 0.0
            for kk in range(2, nz + 1):
                i_q = 1.0 / tke[kk]
                i_ld2[kk] = (n2[kk] * i_lambda2 + f2) * i_q + i_l2_bdry[kk]
                kap_src = h_int[kk] * (k_src[kk] - i_ld2[kk] * kappa[kk]) \
                    + idz[kk - 1] * (kappa[kk - 1] - kappa[kk]) \
                    - idz[kk] * (kappa[kk] - kappa[kk + 1])
                decay_term_k = -idz[kk - 1] * dQmdK[kk] * dKdQ[kk - 1] + h_int[kk] * i_ld2[kk]
                if decay_term_k < 0.0:
                    abort_newton = True
                    break
                bK = 1.0 / (idz[kk] + idz[kk - 1] * cKcomp + decay_term_k)
                cK[kk + 1] = bK * idz[kk]
                cKcomp = bK * (idz[kk - 1] * cKcomp + decay_term_k)
                dKdQ[kk] = bK * (idz[kk - 1] * dKdQ[kk - 1] * cQ[kk]
                                 + dz_int[kk] * (n2[kk] * i_lambda2 + f2) * (i_q * i_q) * kappa[kk])
                dK[kk] = bK * (kap_src + idz[kk - 1] * dK[kk - 1] + idz[kk - 1] * dKdQ[kk - 1] * dQ[kk - 1])
                if dK[kk] <= cKcomp * (kappa_trunc - kappa[kk]):
                    dK[kk] = -cKcomp * kappa[kk]
                elif dK[kk] < cKcomp * (2.0 * kappa_trunc - kappa[kk]):
                    dK[kk] = 2.0 * dK[kk] - cKcomp * (2.0 * kappa_trunc - kappa[kk])

                aQ[kk] = (0.5 * (kappa[kk] + kappa[kk + 1]) + kappa0) * idz[kk]
                dQdz[kk] = 0.5 * (tke[kk] - tke[kk + 1]) * idz[kk]
                tke_src = h_int[kk] * (((kappa[kk] + kappa0) * s2[kk] - kappa[kk] * n2[kk])
                                       - (tke[kk] - q0) * tke_decay[kk]) \
                    - (aQ[kk] * (tke[kk] - tke[kk + 1]) - aQ[kk - 1] * (tke[kk - 1] - tke[kk]))
                v1 = aQ[kk - 1] + dQdz[kk - 1] * dKdQ[kk - 1]
                v2 = (v1 * dQmdK[kk] + dQdz[kk - 1] * cK[kk]) \
                    + ((dQdz[kk - 1] - dQdz[kk]) + dz_int[kk] * (s2[kk] - n2[kk]))
                decay_term_q = h_int[kk] * tke_decay[kk] - dQdz[kk - 1] * dKdQ[kk - 1] * cQ[kk] \
                    - v2 * dKdQ[kk]
                if decay_term_q < 0.0:
                    abort_newton = True
                    break
                bQ = 1.0 / (aQ[kk] + (cQcomp * aQ[kk - 1] + decay_term_q))
                cQ[kk + 1] = aQ[kk] * bQ
                cQcomp = (cQcomp * aQ[kk - 1] + decay_term_q) * bQ
                dQmdK[kk + 1] = (v2 * cK[kk + 1] - dQdz[kk]) * bQ
                dQ[kk] = max(bQ * ((v1 * dQ[kk - 1] + dQdz[kk - 1] * dK[kk - 1])
                                   + (v2 * dK[kk] + tke_src)), cQcomp * (-0.5 * tke[kk]))
                if itt > 1 and kk > ke_src and dK[kk] == 0.0 and (kappa[kk] + kappa[kk + 1]) == 0.0:
                    ke_kappa = kk - 1
                    break
            if ke_kappa == nz and not abort_newton:
                dK[nz + 1] = 0.0
                dKdQ[nz + 1] = 0.0
                dQ[nz + 1] = 0.0
            elif not abort_newton:
                dQ[ke_kappa + 1] = dQ[ke_kappa + 1] / (1.0 - cQ[ke_kappa + 2] * e1[ke_kappa + 2])
                tke[ke_kappa + 1] = max(tke[ke_kappa + 1] + dQ[ke_kappa + 1], tke_min)
                for kk in range(ke_kappa + 2, nz + 2):
                    dK[kk] = 0.0
                    dQ[kk] = max(e1[kk] * dQ[kk - 1], -0.5 * tke[kk])
                    tke[kk] = max(tke[kk] + dQ[kk], tke_min)
                    if abs(dQ[kk]) < ROUNDOFF * tke[kk]:
                        break
            if not abort_newton:
                for kk in range(ke_kappa, 1, -1):
                    dQ[kk] = max(dQ[kk] + (cQ[kk + 1] * dQ[kk + 1] + dQmdK[kk + 1] * dK[kk + 1]),
                                 -0.5 * tke[kk])
                    tke[kk] = max(tke[kk] + dQ[kk], tke_min)
                    dK[kk] = dK[kk] + (cK[kk + 1] * dK[kk + 1] + dKdQ[kk] * dQ[kk])
                    if dK[kk] <= kappa_trunc - kappa[kk]:
                        dK[kk] = -kappa[kk]
                        kappa[kk] = 0.0
                        if kk < ks_src and kk + 1 > ks_kappa:
                            ks_kappa = kk + 1
                    elif dK[kk] < 2.0 * kappa_trunc - kappa[kk]:
                        dK[kk] = 2.0 * dK[kk] - (2.0 * kappa_trunc - kappa[kk])
                        kappa[kk] = max(kappa[kk] + dK[kk], 0.0)
                        if kk <= ks_kappa:
                            ks_kappa = 2
                    else:
                        kappa[kk] = kappa[kk] + dK[kk]
                        if kk <= ks_kappa:
                            ks_kappa = 2
                dQ[1] = max(dQ[1] + cQ[2] * dQ[2] + dQmdK[2] * dK[2], tke_min - tke[1])
                tke[1] = max(tke[1] + dQ[1], tke_min)
                dK[1] = 0.0

        k_lo = min(ks_kappa, ks_kappa_prev)
        k_hi = max(ke_kappa, ke_kappa_prev)
        if tol_err < newton_err and not abort_newton:
            newton_test = 2.0 * newton_err if do_newton else newton_err
            within_tolerance = True
            do_newton = True
            for kk in range(k_lo, k_hi + 1):
                kappa_mean = kappa0 + (kappa[kk] - 0.5 * dK[kk])
                if abs(dK[kk]) > newton_test * kappa_mean:
                    if do_newton:
                        abort_newton = True
                    within_tolerance = False
                    do_newton = False
                    break
                elif abs(dK[kk]) > tol_err * kappa_mean:
                    within_tolerance = False
                    if not do_newton:
                        break
                if abs(dQ[kk]) > newton_test * (tke[kk] - 0.5 * dQ[kk]):
                    if do_newton:
                        abort_newton = True
                    do_newton = False
                    if not within_tolerance:
                        break
        else:
            within_tolerance = True
            for kk in range(k_lo, k_hi + 1):
                if abs(dK[kk]) > tol_err * (kappa0 + (kappa[kk] - 0.5 * dK[kk])):
                    within_tolerance = False
                    break

        if abort_newton:
            do_newton = False
            abort_newton = False
            newton_err = 0.5 * newton_err
            ke_kappa_prev = nz
            for kk in range(2, nz + 1):
                k_q[kk] = kappa[kk] / max(tke[kk], tke_min)

        if within_tolerance:
            break

    if do_newton:
        for kk in range(1, ks_kappa):
            k_q[kk] = 0.0
        for kk in range(ks_kappa, ke_kappa + 1):
            k_q[kk] = kappa[kk] / tke[kk]
        for kk in range(ke_kappa + 1, nz + 2):
            k_q[kk] = 0.0

    for kk in range(2, nz + 1):
        diffusive_src = idz[kk - 1] * (kappa[kk - 1] - kappa[kk]) + idz[kk] * (kappa[kk + 1] - kappa[kk])
        chg_by_k0 = kappa0 * ((idz[kk - 1] + idz[kk]) / h_int[kk] + i_ld2[kk])
        if diffusive_src <= 0.0:
            local_src[kk] = k_src[kk] + chg_by_k0
        else:
            local_src[kk] = (k_src[kk] + chg_by_k0) + diffusive_src / h_int[kk]
    for kk in range(2, nz + 1):
        kappa_src_out[kk] = k_src[kk]
    return tke, kappa, kappa_src_out, local_src


@njit(cache=True)
def _valid_step_nb(ks_kappa, ke_kappa, kappa_src, idtt, nz, n2, s2, tol_max, tol_min, tol_chg, prm):
    """``valid_step`` of ``kappa_shear_column`` (MOM_kappa_shear.F90:1240-1263)."""
    ri_crit = prm[_RI_CRIT]
    for kk in range(max(ks_kappa - 1, 2), min(ke_kappa + 1, nz) + 1):
        if n2[kk] < ri_crit * s2[kk]:
            ksrc = _k_source_nb(n2[kk], s2[kk], prm)
            if (ksrc > max(tol_max[kk], kappa_src[kk] + idtt * tol_chg[kk])) or \
                    (ksrc < min(tol_min[kk], kappa_src[kk] - idtt * tol_chg[kk])):
                return False
        else:
            if 0.0 < min(tol_min[kk], kappa_src[kk] - idtt * tol_chg[kk]):
                return False
    return True


@njit(cache=True)
def _kappa_shear_column_nb(u, v, b, dz, f2, dt, kappa0, prm, kappa_out_m, tke_out_m, j, kc, kf, c):
    """
    ``kappa_shear_column`` (MOM_kappa_shear.F90:866-1374) on the merged column u, v, b, dz (c,) (bed first), same
    operation order; the result is written to column j of kappa_out_m, tke_out_m (N+1, M) through kc, kf.
    """
    nz = c
    n1 = nz + 2
    hlay = np.zeros(n1)
    u0xdz = np.zeros(n1)
    v0xdz = np.zeros(n1)
    b0xdz = np.zeros(n1)
    for k in range(nz):
        hlay[nz - k] = dz[k]
    for k in range(nz):
        u0xdz[nz - k] = u[k] * hlay[nz - k]
        v0xdz[nz - k] = v[k] * hlay[nz - k]
        b0xdz[nz - k] = b[k] * hlay[nz - k]

    max_ks_it = int(prm[_MAX_KS_IT])
    tol_err = prm[_TOL_ERR]
    k0dt = dt * kappa0
    kappa_trunc = prm[_KD_TRUNC_FRAC] * kappa0
    lz = prm[_LZ_RESCALE]
    i_lz_rescale_sqr = 1.0 / (lz * lz) if lz > 0 else 1.0
    tol_dksrc = prm[_MAX_KAP_SRC_CHG]
    tol_dksrc_low = 0.95 if tol_dksrc == 10.0 else (tol_dksrc - 0.5) / tol_dksrc
    tol2 = 2.0 * tol_err

    idz = np.zeros(n1)
    for k in range(1, nz + 1):
        idz[k] = 1.0 / hlay[k]
    i_dz_int = np.zeros(n1)
    dist_from_top = np.zeros(n1)
    i_dz_int[1] = 2.0 / hlay[1]
    for kk in range(2, nz + 1):
        i_dz_int[kk] = 2.0 / (hlay[kk - 1] + hlay[kk])
        dist_from_top[kk] = dist_from_top[kk - 1] + hlay[kk - 1]
    i_dz_int[nz + 1] = 2.0 / hlay[nz]
    i_l2_bdry = np.zeros(n1)
    dist_from_bot = 0.0
    for kk in range(nz, 1, -1):
        dist_from_bot += hlay[kk]
        dt_ = dist_from_top[kk]
        i_l2_bdry[kk] = ((dt_ + dist_from_bot) * (dt_ + dist_from_bot)) / \
            ((dt_ * dist_from_bot) * (dt_ * dist_from_bot))
        i_l2_bdry[kk] = i_lz_rescale_sqr * i_l2_bdry[kk]

    uu = np.zeros(n1)
    vv = np.zeros(n1)
    bb = np.zeros(n1)
    a1 = np.zeros(n1)
    c1 = np.zeros(n1)
    a1[2] = k0dt * i_dz_int[2]
    b1 = 1.0 / (hlay[1] + a1[2])
    uu[1] = b1 * u0xdz[1]
    vv[1] = b1 * v0xdz[1]
    bb[1] = b1 * b0xdz[1]
    c1[2] = a1[2] * b1
    d1 = hlay[1] * b1
    for kk in range(2, nz):
        bd1 = hlay[kk] + d1 * a1[kk]
        a1[kk + 1] = k0dt * i_dz_int[kk + 1]
        b1 = 1.0 / (bd1 + a1[kk + 1])
        uu[kk] = b1 * (u0xdz[kk] + a1[kk] * uu[kk - 1])
        vv[kk] = b1 * (v0xdz[kk] + a1[kk] * vv[kk - 1])
        bb[kk] = b1 * (b0xdz[kk] + a1[kk] * bb[kk - 1])
        c1[kk + 1] = a1[kk + 1] * b1
        d1 = bd1 * b1
    b1 = 1.0 / ((hlay[nz] + d1 * a1[nz]) + k0dt * i_dz_int[nz + 1])
    uu[nz] = b1 * (u0xdz[nz] + a1[nz] * uu[nz - 1])
    vv[nz] = b1 * (v0xdz[nz] + a1[nz] * vv[nz - 1])
    b1 = 1.0 / (hlay[nz] + d1 * a1[nz])
    bb[nz] = b1 * (b0xdz[nz] + a1[nz] * bb[nz - 1])
    for kk in range(nz - 1, 0, -1):
        uu[kk] = uu[kk] + c1[kk + 1] * uu[kk + 1]
        vv[kk] = vv[kk] + c1[kk + 1] * vv[kk + 1]
        bb[kk] = bb[kk] + c1[kk + 1] * bb[kk + 1]

    h_int = np.zeros(n1)
    h_int[2] = hlay[1]
    for kk in range(2, nz):
        norm = 1.0 / (hlay[kk] * (hlay[kk - 1] + hlay[kk + 1]) + 2.0 * hlay[kk - 1] * hlay[kk + 1])
        wt_a = ((hlay[kk] + hlay[kk + 1]) * hlay[kk - 1]) * norm
        wt_b = ((hlay[kk - 1] + hlay[kk]) * hlay[kk + 1]) * norm
        h_int[kk] = h_int[kk] + hlay[kk] * wt_a
        h_int[kk + 1] = hlay[kk] * wt_b
    h_int[nz] = h_int[nz] + hlay[nz]
    h_int[nz + 1] = 0.0
    dz_int = h_int

    n2 = np.zeros(n1)
    s2 = np.zeros(n1)
    kappa = np.zeros(n1)
    for k in range(1, nz + 2):
        kappa[k] = prm[_KD_SEED]
    _projected_state_nb(kappa, uu, vv, bb, 0.0, nz, hlay, i_dz_int, uu, vv, bb, n2, s2, -1, -1)

    dt_rem = dt
    k_q = np.zeros(n1)
    kappa_avg = np.zeros(n1)
    tke_avg = np.zeros(n1)
    local_src_avg = np.zeros(n1)
    for kk in range(1, nz + 2):
        if h_int[kk] > 0.0:
            local_src_avg[kk] = 0.1 * k0dt * i_dz_int[kk] / h_int[kk]
    u_test = np.zeros(n1)
    v_test = np.zeros(n1)
    b_test = np.zeros(n1)
    kappa_mid = np.zeros(n1)
    tol_max = np.zeros(n1)
    tol_min = np.zeros(n1)
    tol_chg = np.zeros(n1)

    for itt in range(1, max_ks_it + 1):
        tke, kappa_out, kappa_src, local_src = _find_kappa_tke_nb(
            n2, s2, kappa, idz, h_int, dz_int, i_l2_bdry, f2, nz, prm, kappa0, kappa_trunc, k_q)

        ks_kappa, ke_kappa = nz + 1, 0
        for kk in range(2, nz + 1):
            if kappa_out[kk] > 0.0:
                ks_kappa = kk
                break
        for kk in range(nz, ks_kappa - 1, -1):
            if kappa_out[kk] > 0.0:
                ke_kappa = kk
                break
        if ke_kappa == nz:
            kappa_out[nz + 1] = 0.0

        if ke_kappa < ks_kappa or itt == max_ks_it:
            dt_now = dt_rem
        else:
            dt_test = dt_rem
            for kk in range(2, nz + 1):
                tol_max[kk] = kappa_src[kk] + tol_dksrc * local_src[kk]
                tol_min[kk] = kappa_src[kk] - tol_dksrc_low * local_src[kk]
                tol_chg[kk] = tol2 * local_src_avg[kk]
            valid_dt = False
            for _ in range((max_ks_it + 1 - itt) // 2):
                _projected_state_nb(kappa_out, uu, vv, bb, 0.5 * dt_test, nz, hlay, i_dz_int,
                                    u_test, v_test, b_test, n2, s2, ks_kappa, ke_kappa)
                valid_dt = _valid_step_nb(ks_kappa, ke_kappa, kappa_src, 1.0 / dt_test, nz, n2, s2,
                                          tol_max, tol_min, tol_chg, prm)
                if valid_dt:
                    break
                dt_test = 0.5 * dt_test
            if dt_test < dt_rem and valid_dt:
                dt_inc = 0.5 * dt_test
                for _ in range(DT_REFINEMENTS):
                    _projected_state_nb(kappa_out, uu, vv, bb, 0.5 * (dt_test + dt_inc), nz, hlay,
                                        i_dz_int, u_test, v_test, b_test, n2, s2, ks_kappa, ke_kappa)
                    valid_dt = _valid_step_nb(ks_kappa, ke_kappa, kappa_src, 1.0 / (dt_test + dt_inc), nz,
                                              n2, s2, tol_max, tol_min, tol_chg, prm)
                    if valid_dt:
                        dt_test = dt_test + dt_inc
                    dt_inc = 0.5 * dt_inc
            else:
                dt_inc = 0.0
            dt_now = min(dt_test * (1.0 + tol_err) + dt_inc, dt_rem)
            for kk in range(2, nz + 1):
                local_src_avg[kk] += dt_now * local_src[kk]

        if ke_kappa < ks_kappa:
            dt_wt = dt_rem / dt
            dt_rem = 0.0
            kappa_mid[:] = 0.0
            for kk in range(1, nz + 2):
                tke_avg[kk] += dt_wt * tke[kk]
        else:
            _projected_state_nb(kappa_out, uu, vv, bb, dt_now, nz, hlay, i_dz_int,
                                u_test, v_test, b_test, n2, s2, ks_kappa, ke_kappa)
            k_q_tmp = k_q.copy()
            tke_pred, kappa_pred, _a, _b = _find_kappa_tke_nb(
                n2, s2, kappa_out, idz, h_int, dz_int, i_l2_bdry, f2, nz, prm, kappa0, kappa_trunc, k_q_tmp)
            ks_kappa, ke_kappa = nz + 1, 0
            for kk in range(1, nz + 2):
                kappa_mid[kk] = 0.5 * (kappa_out[kk] + kappa_pred[kk])
                if kappa_mid[kk] > 0.0 and kk < ks_kappa:
                    ks_kappa = kk
                if kappa_mid[kk] > 0.0:
                    ke_kappa = kk
            _projected_state_nb(kappa_mid, uu, vv, bb, dt_now, nz, hlay, i_dz_int,
                                u_test, v_test, b_test, n2, s2, ks_kappa, ke_kappa)
            tke_pred, kappa_pred, _a, _b = _find_kappa_tke_nb(
                n2, s2, kappa_out, idz, h_int, dz_int, i_l2_bdry, f2, nz, prm, kappa0, kappa_trunc, k_q)
            dt_wt = dt_now / dt
            dt_rem = dt_rem - dt_now
            for kk in range(1, nz + 2):
                kappa_mid[kk] = 0.5 * (kappa_out[kk] + kappa_pred[kk])
                kappa_avg[kk] += kappa_mid[kk] * dt_wt
                tke_avg[kk] += dt_wt * 0.5 * (tke_pred[kk] + tke[kk])
            kappa = kappa_pred.copy()

        if dt_rem > 0.0:
            _projected_state_nb(kappa_mid, uu, vv, bb, dt_now, nz, hlay, i_dz_int,
                                uu, vv, bb, n2, s2, -1, -1)
        if dt_rem <= 0.0:
            break

    # Back to the model orientation (merged interface q = top-down index nz + 1 − q), then to the model interfaces.
    n_if = kc.shape[0]
    for i in range(n_if):
        q_lo = kc[i]
        q_hi = min(kc[i] + 1, c)
        lo_k, hi_k = kappa_avg[nz + 1 - q_lo], kappa_avg[nz + 1 - q_hi]
        lo_q, hi_q = tke_avg[nz + 1 - q_lo], tke_avg[nz + 1 - q_hi]
        f = kf[i]
        if f == 0.0:
            kappa_out_m[i, j] = lo_k
            tke_out_m[i, j] = lo_q
        elif f == 1.0:
            kappa_out_m[i, j] = hi_k
            tke_out_m[i, j] = hi_q
        else:
            kappa_out_m[i, j] = (1.0 - f) * lo_k + f * hi_k
            tke_out_m[i, j] = (1.0 - f) * lo_q + f * hi_q


@njit(cache=True)
def _local_buoyancy(ht, hs, hc, c, code, rho0, g, hb):
    """
    Locally referenced buoyancy hb (c,) [m s⁻²] of a merged column (module docstring) from its T, S (c,) and
    thicknesses hc (c,) [m], merged layer 0 at the bed: hb_0 = 0, hb_q = hb_q−1 + Δb at the interface between q−1
    and q (``buoyancy_jump_elem``), at the pressure Σ_{q' ≥ q} g ρ₀ hc_q' [Pa] summed from the surface.
    """
    gr0 = g * rho0
    g_rho0 = g / rho0
    pint = np.zeros(c)
    pq = 0.0
    for q in range(c - 1, 0, -1):
        pq = pq + gr0 * hc[q]
        pint[q] = pq
    hb[0] = 0.0
    for q in range(1, c):
        hb[q] = hb[q - 1] + buoyancy_jump_elem(ht[q], hs[q], ht[q - 1], hs[q - 1], pint[q], code, g_rho0)


@njit(cache=True, parallel=True)
def _columns_nb(u, v, b, dz, f2, dt, kappa0, h_thin, prm, h_massless, kappa_out, tke_out, tem, sal, code, rho0, g):
    """Thin-layer merge + ``_kappa_shear_column_nb`` of every column of dz (N, M), one column per thread task;
    ``code`` > 0 (nonlinear EOS): the locally referenced buoyancy of ``tem``, ``sal`` (``_local_buoyancy``)."""
    n, mcol = dz.shape
    for j in prange(mcol):
        dry = 0.0
        for k in range(n):
            dry += dz[k, j]
        if dry <= h_massless:
            continue
        # ``core.components.closure.active_layers``
        act = np.zeros(n, np.bool_)
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
        # ``_merge_map``
        lay = np.empty(n, np.int64)
        nxt = -1
        for k in range(n - 1, -1, -1):
            if act[k]:
                nxt = k
            lay[k] = nxt
        prv = -1
        rank = np.empty(n, np.int64)
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
        hc0 = np.zeros(c)
        for k in range(n):
            hc0[lay[k]] += dz[k, j]
        kc = np.empty(n + 1, np.int64)
        kf = np.empty(n + 1)
        d_in = 0.0
        for i in range(n):
            kc[i] = lay[i]
            if i == 0 or lay[i] != lay[i - 1]:
                kf[i] = 0.0
                d_in = dz[i, j]
            else:
                kf[i] = d_in / hc0[lay[i]]
                d_in = d_in + dz[i, j]
        kc[n] = lay[n - 1]
        kf[n] = 1.0
        # merged column (kappa_shear_columns_numpy)
        hc = np.zeros(c)
        hu = np.zeros(c)
        hv = np.zeros(c)
        hb = np.zeros(c)
        ht = np.zeros(c)
        hs = np.zeros(c)
        for k in range(n):
            h = dz[k, j]
            hc[lay[k]] += h
            hu[lay[k]] += h * u[k, j]
            hv[lay[k]] += h * v[k, j]
            if code == 0:
                hb[lay[k]] += h * b[k, j]
            else:
                ht[lay[k]] += h * tem[k, j]
                hs[lay[k]] += h * sal[k, j]
        for q in range(c):
            hu[q] = hu[q] / hc[q]
            hv[q] = hv[q] / hc[q]
            if code == 0:
                hb[q] = hb[q] / hc[q]
            else:
                ht[q] = ht[q] / hc[q]
                hs[q] = hs[q] / hc[q]
        if code != 0:
            _local_buoyancy(ht, hs, hc, c, code, rho0, g, hb)
        _kappa_shear_column_nb(hu, hv, hb, hc, f2[j], dt, kappa0, prm, kappa_out, tke_out, j, kc, kf, c)


def kappa_shear_columns(u: np.ndarray, v: np.ndarray, b: np.ndarray, dz: np.ndarray, f2: np.ndarray, dt: float,
                        kappa0: float, h_thin: np.ndarray, p: JacksonParams,
                        h_massless: float = H_MASSLESS, T: np.ndarray | None = None, S: np.ndarray | None = None,
                        eos=None, rho0: float = 1000.0, g: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
    """
    Jackson et al. (2008) κ [m² s⁻¹] and Q [m² s⁻²] of M columns with thin-layer merging, compiled (Numba, threads
    over the columns); bit for bit with ``kappa_shear_columns_numpy`` (same arguments and returns: u, v, b, dz
    (N, M) with k = 0 at the bed, f2 and h_thin (M,), outputs (N+1, M) at the model interfaces; T, S, eos, rho0, g
    for a nonlinear EOS, module docstring).
    """
    code = 0 if eos is None else eos_code(eos)
    if not kappa0 > 0.0:
        raise ValueError("kappa_shear_columns: kappa0 must be > 0 (MOM6 uses max(KD, 1e-7))")
    if not dt > 0.0:
        raise ValueError("kappa_shear_columns: dt must be > 0")
    dz = np.ascontiguousarray(dz, dtype=float)
    if dz.ndim != 2:
        raise ValueError("kappa_shear_columns: dz must be (N, M)")
    n, m = dz.shape
    u, v, b = (np.ascontiguousarray(x, dtype=float) for x in (u, v, b))
    for name, arr in (("u", u), ("v", v), ("b", b)):
        if arr.shape != dz.shape:
            raise ValueError(f"kappa_shear_columns: {name} must have shape {dz.shape}, got {arr.shape}")
    if not np.isfinite(dz).all() or (dz < 0.0).any():
        raise ValueError("kappa_shear_columns: dz must be finite and >= 0 (dry columns: zero thicknesses)")
    f2, h_thin = np.asarray(f2, float), np.asarray(h_thin, float)
    for name, arr in (("f2", f2), ("h_thin", h_thin)):
        if arr.ndim != 0 and arr.shape != (m,):
            raise ValueError(f"kappa_shear_columns: {name} must be a scalar or have shape ({m},) (one value per "
                             f"column of dz), got {arr.shape}")
    f2 = np.ascontiguousarray(np.broadcast_to(f2, (m,)))
    h_thin = np.ascontiguousarray(np.broadcast_to(h_thin, (m,)))
    kappa = np.zeros((n + 1, m))
    tke = np.zeros((n + 1, m))
    if code:
        tem, sal = (np.ascontiguousarray(x, dtype=float) for x in (T, S))
        for name, arr in (("T", tem), ("S", sal)):
            if arr.shape != dz.shape:
                raise ValueError(f"kappa_shear_columns: {name} must have shape {dz.shape}, got {arr.shape}")
    else:
        tem = sal = np.zeros((1, 1))
    _columns_nb(u, v, b, dz, f2, float(dt), float(kappa0), h_thin, params_array(p), float(h_massless), kappa, tke,
                tem, sal, code, float(rho0), float(g))
    return kappa, tke
