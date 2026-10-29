"""
Bottom friction of a depth-averaged model from a logarithmic boundary layer.

A 2-D model sees only the depth-mean velocity ū.  If the whole column is a
logarithmic layer over a bed of roughness length z₀,

    u(z) = (u*/κ) ln(z/z₀),   z₀ < z < D        (law of the wall)

its mean over the depth D is ū = (u*/κ) [ln(D/z₀) − 1 + z₀/D], and the bed
stress τ_b/ρ₀ = u*² = C_d ū² with

    C_d(D) = [ κ / (ln(D/z₀) − 1 + z₀/D) ]²      (Soulsby 1997, eq. 37)

C_d grows as the water shoals: in CROCO's 3-D TIDAL_FLAT (z₀ = 1e-4 m, GLS)
the effective coefficient bustr/(ρ₀ ū²) goes from 2.8e-3 at 0.2–0.5 m to
1.3e-3 at 12–20 m, which this formula follows (3.0e-3 … 1.4e-3).  A constant
C_d therefore over-damps deep water and under-damps the flats.

References
----------
Soulsby, R.L. (1997). Dynamics of Marine Sands. Thomas Telford, London, §3.
"""

from __future__ import annotations

import numpy as np

KAPPA = 0.41          # von Kármán constant (CROCO vonKar)


def log_law_cd(depth: np.ndarray | float, z0: float, kappa: float = KAPPA, cd_min: float = 1e-4,
               cd_max: float = 0.1) -> np.ndarray:
    """
    Depth-dependent quadratic drag coefficient C_d(D) [–] for τ_b/ρ₀ = C_d |ū| ū.

    depth : total water depth D [m] (dry, NaN or D ≤ e·z₀ → cd_max)
    z0 : bed roughness length [m]
    cd_min, cd_max : bounds, as CROCO's Cdb_min / Cdb_max
    """
    D = np.asarray(depth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.log(D / z0) - 1.0 + z0 / D
        cd = (kappa / denom) ** 2
    cd = np.where(np.isfinite(D) & (D > z0) & (denom > 0) & np.isfinite(cd), cd, cd_max)
    return np.clip(cd, cd_min, cd_max)


def first_layer_cd(dz_bottom: np.ndarray | float, z0: float, kappa: float = KAPPA, cd_min: float = 1e-4,
                   cd_max: float = 0.1) -> np.ndarray:
    """
    Drag coefficient of a layered (3-D) model, τ_b/ρ₀ = C_d |u₁| u₁ with u₁ the
    velocity of the bottom layer, taken as the law of the wall at its mid-point:

        C_d = [ κ / ln(z₁/z₀) ]²,   z₁ = Δz₁ / 2        (CROCO set_depth/bottom stress;
                                                         Soulsby 1997 §3)
    dz_bottom : thickness of the bottom layer [m] (≤ 2 z₀ or invalid → cd_max)
    """
    dz = np.asarray(dz_bottom, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cd = (kappa / np.log(0.5 * dz / z0)) ** 2
    cd = np.where(np.isfinite(dz) & (0.5 * dz > z0) & np.isfinite(cd), cd, cd_max)
    return np.clip(cd, cd_min, cd_max)


def cross_velocity_at_faces(ub: np.ndarray, vb: np.ndarray, periodic_x: bool = False,
                            periodic_y: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    The other velocity component at the faces of an Arakawa C grid, for the full bed speed of a quadratic drag
    |U| = √(u² + v̄²) at u-faces and √(v² + ū²) at v-faces.

        v̄_{j,i−½} = ¼ (v_{j−½,i−1} + v_{j+½,i−1} + v_{j−½,i} + v_{j+½,i})      (CROCO get_vbc.F:517-518, :525-526)

    computed as the mean of the two cell-centre values ½ (v_{j−½} + v_{j+½}), so the arithmetic is that of the
    ALE quadratic drag (``core.components.bottom_drag.QuadraticBedDrag``, bit for bit).  MOM6 weights the four
    faces by their thickness (``set_v_at_u``, MOM_set_viscosity.F90:1887-1929); with the bed-layer velocities of a
    log-law drag CROCO's plain mean is used.

    ub : (ny, nx+1) u at the u-faces [m s⁻¹];  vb : (ny+1, nx) v at the v-faces [m s⁻¹]
    periodic_x, periodic_y : periodic axes; the wrap faces (u-faces 0 and nx, v-faces 0 and ny) average the cells
        n−1 and 0.  On a closed edge the face is a wall (u = 0 there, no stress), the cross component is set to 0.
    Returns (v_at_u (ny, nx+1), u_at_v (ny+1, nx)) [m s⁻¹].
    """
    vc = 0.5 * (vb[:-1] + vb[1:])                       # v at the cell centres (ny, nx)
    uc = 0.5 * (ub[:, :-1] + ub[:, 1:])                 # u at the cell centres (ny, nx)
    v_at_u = np.zeros_like(ub)
    v_at_u[:, 1:-1] = 0.5 * (vc[:, :-1] + vc[:, 1:])
    u_at_v = np.zeros_like(vb)
    u_at_v[1:-1, :] = 0.5 * (uc[:-1, :] + uc[1:, :])
    if periodic_x:                                      # wrap faces: cells nx−1 and 0
        v_at_u[:, 0] = v_at_u[:, -1] = 0.5 * (vc[:, -1] + vc[:, 0])
    if periodic_y:
        u_at_v[0, :] = u_at_v[-1, :] = 0.5 * (uc[-1, :] + uc[0, :])
    return v_at_u, u_at_v
