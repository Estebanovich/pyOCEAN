"""
Real freshwater flux in the tracers (N2a, spec docs/superpowers/specs/2026-10-28-n2a-salinity-freshwater-design.md
§3): the substep that follows the advection and precedes the vertical diffusion.

The surface freshwater flux fw = P − E [m s⁻¹, positive into the ocean] is a volume source of the free surface
(Huang 1993; MITgcm ``useRealFreshWaterFlux``, integr_continuity.F:93,185).  For the tracers the freshwater
carries S = 0 and the temperature of the top layer, so in each cell-layer whose volume changes from V_adv (after
the advection, without freshwater) to V_fw:

    salt-like members (S, S², passive tracers):  c_new = c V_adv / V_fw    (content V c unchanged)
    T and T²:                                     unchanged               (the water enters/leaves at the top T)

as MOM6 applyBoundaryFluxesInOut (src/parameterizations/vertical/MOM_diabatic_aux.F90:1013-1117, MOM6 main of
2025): incoming mass with ``Temp_in = T2d(i,1)``, ``Salin_in = 0`` (l. 1034-1035), outgoing mass at the layer T and
S = 0, so S rises by h_old/h_new.

Evaporation that would take the top layer below ``floor`` (``core.vertical.ale.DZ_FLOOR`` per unit area) takes the
rest from the layers below, top-down, at fixed salt — MOM6 does the same with a CFL fraction ``evap_CFL_limit``
of each layer (MOM_diabatic_aux.F90:1093-1117); here each layer keeps the floor, and only when the whole column
holds less than its floors are the layers taken down to zero (the free-surface positivity guard has already
clipped such a column, ``BarotropicFluxes.clip_volume``).  The columns concerned are counted (``n_limited``),
never silently.

Index order (k, j, i), k = 0 at the bed: the top layer is k = N−1.
"""
from __future__ import annotations

import numpy as np


def freshwater_volumes(V: np.ndarray, dV: np.ndarray, floor: np.ndarray | float) -> tuple[np.ndarray, int]:
    """
    Add the freshwater volume ``dV`` (ny, nx) to the top layer of the layer volumes (or thicknesses) ``V``
    (N, ny, nx).  A removal (dV < 0) that would take the top layer below ``floor`` (same units as V) is taken from
    the layers below, top-down, each kept at ``floor``; a deficit left after that empties the layers down to 0, so
    the result is ≥ 0 and Σ_k V_new = Σ_k V + dV whenever the column holds enough water.

    Returns (V_new, n_limited): ``n_limited`` = number of columns whose deficit spilled below the top layer.
    Units are those of ``V`` (m³ with dV = Δt A fw, or m with dV = Δt fw).
    """
    out = V.copy()
    out[-1] += dV
    floor = np.broadcast_to(np.asarray(floor, float), dV.shape)
    spill = (dV < 0) & (out[-1] < floor)
    if not spill.any():
        return out, 0
    js, is_ = np.nonzero(spill)
    col = V[:, js, is_].copy()                                       # (N, m) spilling columns, before the source
    deficit = -dV[js, is_]
    fl = floor[js, is_]
    n = V.shape[0]
    for keep in (fl, np.zeros_like(fl)):                             # first keep the floor, then down to zero
        for k in range(n - 1, -1, -1):
            take = np.minimum(deficit, np.maximum(col[k] - keep, 0.0))
            col[k] -= take
            deficit = deficit - take
    out[:, js, is_] = col
    return out, int(spill.sum())


def apply_freshwater(stack: np.ndarray, layout: dict[str, int], V_adv: np.ndarray, V_fw: np.ndarray,
                     fw_area_dt: np.ndarray, *, dt: float) -> tuple[np.ndarray, int, np.ndarray | None]:
    """
    Freshwater substep on the advected tracer stack (in place, also returned).

    stack       (n_stack, N, ny, nx) in the ``_stack_layout`` order ``[T, (T²), S, (S²), tr…]``
    layout      positions from ``core.model.driver._stack_layout``
    V_adv       layer volumes after the advection, without freshwater (N, ny, nx) [m³]
    V_fw        layer volumes after the freshwater, from ``freshwater_volumes`` (N, ny, nx) [m³]
    fw_area_dt  Δt A fw (ny, nx) [m³]: columns with a nonzero value are the only ones touched
    dt          time step [s]

    S, S² and the passive tracers: c ← c V_adv / V_fw (salt content kept; freshwater has S = 0); T and T²
    unchanged (the freshwater has the top-layer T).  In a column where the change reached below the top layer
    (evaporation deficit, or in σ rain beyond a thin top layer) the touched layers share c = Σ V_adv c / Σ V_fw.
    A column emptied to V_fw = 0 keeps its values.

    Returns (stack, n_limited, var_surf):
      n_limited  columns where the volume change reached below the top layer (evaporation deficit, see
                 ``freshwater_volumes``);
      var_surf   (N, ny, nx) [(g kg⁻¹)² m³ s⁻¹] the salinity-variance source (V_fw S_new² − V_adv S_old²)/Δt of the
                 freshwater (Lorenz et al. 2021 eq. 43 [verificar]); zero where no freshwater; None without S.
    """
    cols = fw_area_dt != 0
    ratio = np.ones_like(V_adv)
    np.divide(V_adv, V_fw, out=ratio, where=cols[None] & (V_fw > 0))
    var_surf = None
    if "S" in layout:
        S_old = stack[layout["S"]].copy()
    salt_like = [i for i in range(layout["n"]) if i not in (layout["T"], layout.get("T2"))]
    spill = (V_adv[:-1] != V_fw[:-1]).any(axis=0)
    if spill.any():
        # columns whose volume change reached below the top layer: the layers touched (top down to the lowest
        # changed one) share one concentration Σ V_adv c / Σ V_fw, so the salt of a layer held at the floor does
        # not stay behind at an extreme concentration (≈ c h†/DZ_FLOOR, which the remap's round-off in Σ h would
        # amplify); the column content is unchanged
        changed = V_adv != V_fw
        kmin = np.argmax(changed, axis=0)                              # lowest changed layer (k = 0 at the bed)
        below = np.arange(V_adv.shape[0])[:, None, None] >= kmin[None]
        touched = below & spill[None]
        Vf_sum = np.where(touched, V_fw, 0.0).sum(axis=0)
        has_water = Vf_sum > 0
    n_limited = int(np.count_nonzero(spill))
    for i in salt_like:
        c = np.where(cols[None], stack[i] * ratio, stack[i])
        if spill.any():
            cbar = np.divide(np.where(touched, V_adv * stack[i], 0.0).sum(axis=0), Vf_sum,
                             out=np.zeros_like(Vf_sum), where=has_water)
            c = np.where(touched & has_water[None], cbar[None], c)
        stack[i] = c
    if "S" in layout:
        S_new = stack[layout["S"]]
        var_surf = np.where(cols[None], (V_fw * S_new**2 - V_adv * S_old**2) / dt, 0.0)
    return stack, n_limited, var_surf
