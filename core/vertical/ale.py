"""
Layer geometry of the ALE vertical coordinate on the Arakawa C grid (spec
docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md §2, §5).

Convention: layer thickness h (N, ny, nx) [m], k = 0 at the bed; u-faces (N, ny, nx+1),
v-faces (N, ny+1, nx).

Three face thicknesses, each for its job (ALE Plan B2, 2026-10-12):

* Transport (``face_transport_thickness``, ``upwind_transport_x``): the thickness of the cell the
  water comes from,

      Q_k = L u_k q_k ,   q_k = h_k,up  (clipped at the shallower bed for z* and hybrid),

  so a thick layer can flow into a cell where it has vanished — an isopycnal layer that
  outcrops, the lock-exchange front — while water facing rock does not move (HYCOM
  cnuity.F90:252-256).  The layer fluxes are then corrected to the volume flux of
  the free-surface solve, sharing the residual in proportion to the layers of its upwind cell
  (HYCOM cnuity.F90:585-620, "upstream barotropic correction"; MOM6 reaches the same constraint by
  iterating a uniform velocity correction, MOM_continuity_PPM.F90:1368-1546, Hallberg & Adcroft
  2009).  Σ_k Q_k is then exactly the free-surface flux, so Σ_k h_k = H + η.  Donor cell is
  first order; MOM6 uses PPM with a positive-definite limiter (Adcroft et al. 2019, §A) — the
  upgrade path if a benchmark shows the difference.  σ-layers are not clipped: they follow the
  terrain, and layer k of the deep column continues as layer k of the shallow one.  Without the
  clip in z*, the residual pressure-gradient error of a layer facing rock accelerates it with
  nothing to oppose it (seamount: 0.02 → 35 cm/s in one day).
* Mean (q_E + q_W)/2, direction independent, for the depth mean of the free surface
  ū = Σ w_k u_k, the vertical viscosity and the thin-bed rule.  A layer that has vanished on one
  side keeps half the other side's thickness; a z* layer facing rock is thin and carries no
  velocity; a partial z* layer gets the part above the shallow bed (MITgcm hFacW).  For σ-layers
  w_k = Δσ_k.
* Min (``face_thickness``): min(h_L, h_R), as MITgcm hFacW (model/src/ini_masks_etc.F:250-252),
  for the along-layer tracer diffusion, so no diffusive flux enters a vanished cell.

References: Adcroft, A. & Hallberg, R. (2006). Ocean Modelling 11, 224–233.  Hallberg, R. &
Adcroft, A. (2009). Ocean Modelling 29, 15–26 [verificar].  Adcroft, A. et al. (2019). JAMES 11,
3167–3211.  Bleck, R. (2002). Ocean Modelling 4, 55–88.  HYCOM hybgen.F90:1757-1764 (thin
near-bottom layers).
"""

from __future__ import annotations

import numpy as np

H_THIN_BED = 0.01      # [m] bed layers thinner than this carry no velocity (HYCOM hybgen.F90:1757-1764)
DZ_FLOOR = 1e-6        # [m] smallest thickness passed to the implicit vertical solvers


def face_thickness(h: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Layer thickness at u- and v-faces [m]: min(h_L, h_R) between two active cells, the inner
    cell on the domain edge, 0 next to land.  h (N, ny, nx), active (ny, nx) bool.
    """
    h = np.where(active[None], np.nan_to_num(h), 0.0)
    n, ny, nx = h.shape
    hu = np.zeros((n, ny, nx + 1))
    hv = np.zeros((n, ny + 1, nx))
    hu[:, :, 1:-1] = np.where((active[:, :-1] & active[:, 1:])[None], np.minimum(h[:, :, :-1], h[:, :, 1:]), 0.0)
    hu[:, :, 0], hu[:, :, -1] = h[:, :, 0], h[:, :, -1]
    hv[:, 1:-1, :] = np.where((active[:-1, :] & active[1:, :])[None], np.minimum(h[:, :-1, :], h[:, 1:, :]), 0.0)
    hv[:, 0, :], hv[:, -1, :] = h[:, 0, :], h[:, -1, :]
    return hu, hv


def face_transport_thickness(h: np.ndarray, H: np.ndarray, active: np.ndarray,
                             clip: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Layer thickness available for transport through the u-faces [m]: q_E (eastward, from the
    west cell) and q_W (westward, from the east cell), each (N, ny, nx+1).  h (N, ny, nx) bed
    first, H (ny, nx) rest depth [m], active (ny, nx).  With ``clip`` (z*, hybrid) only the part of
    the upwind layer above the shallower bed passes (HYCOM cnuity.F90:252-256):

        q = min(h_k,up, max(0, z_top,k,up − max(−H_W, −H_E)))

    Without it (σ: the layers follow the terrain) q = h_k,up.  Domain-edge faces use the inner
    cell; faces next to land are 0.  For the v-faces call it on swapped axes.
    """
    h = np.where(active[None], np.nan_to_num(h), 0.0)
    Hn = np.nan_to_num(np.asarray(H, float))
    hW = np.concatenate([h[..., :1], h], axis=-1)                    # cell west of each face
    hE = np.concatenate([h, h[..., -1:]], axis=-1)                   # cell east of each face
    aW = np.concatenate([active[:, :1], active], axis=-1)
    aE = np.concatenate([active, active[:, -1:]], axis=-1)
    open_ = (aW & aE)[None]
    if clip:
        HW = np.concatenate([Hn[:, :1], Hn], axis=-1)
        HE = np.concatenate([Hn, Hn[:, -1:]], axis=-1)
        z_bed = np.maximum(-HW, -HE)[None]                             # the shallower bed
        topW = -HW[None] + np.cumsum(hW, axis=0)
        topE = -HE[None] + np.cumsum(hE, axis=0)
        hW = np.minimum(hW, np.maximum(0.0, topW - z_bed))
        hE = np.minimum(hE, np.maximum(0.0, topE - z_bed))
    return np.where(open_, hW, 0.0), np.where(open_, hE, 0.0)


def upwind_transport_x(q_e: np.ndarray, q_w: np.ndarray, ut: np.ndarray, L: np.ndarray,
                       Q_bt: np.ndarray) -> np.ndarray:
    """
    Layer volume fluxes through the u-faces [m³ s⁻¹], donor cell and consistent with the free surface:

        Q_k = L u_k q_k(u_k) + R q_k(R) / Σ_j q_j(R) ,    R = Q_bt − Σ_k L u_k q_k(u_k)

    q_e, q_w (N, ny, nx+1) from ``face_transport_thickness``; q(s) is q_e for s ≥ 0, q_w
    otherwise (the other one if that column is empty).  ut (N, ny, nx+1) transport velocity
    [m s⁻¹], L (ny, nx+1) face length [m], Q_bt (ny, nx+1) volume flux of the free-surface solve
    [m³ s⁻¹].  For the v-faces call it on swapped axes.
    """
    F = L[None] * ut * np.where(ut >= 0, q_e, q_w)
    R = Q_bt - F.sum(axis=0)
    qR = np.where((R >= 0)[None], q_e, q_w)
    alt = np.where((R >= 0)[None], q_w, q_e)
    qR = np.where((qR.sum(axis=0) > 0)[None], qR, alt)
    tot = qR.sum(axis=0)
    return F + R[None] * qR / np.where(tot > 0, tot, 1.0)[None]


def face_fractions(hf: np.ndarray) -> np.ndarray:
    """w_k = h_k,f / Σ_k h_k,f (uniform 1/N on faces with no water, so ū stays defined)."""
    tot = hf.sum(axis=0)
    wet = tot > 0
    return np.where(wet[None], hf / np.where(wet, tot, 1.0)[None], 1.0 / hf.shape[0])


def zero_thin_bed_velocity(u: np.ndarray, hf: np.ndarray, w: np.ndarray, h_thin: float = H_THIN_BED) -> np.ndarray:
    """
    Zero velocity in the run of face layers thinner than ``h_thin`` that starts at the bed
    (HYCOM hybgen.F90:1757-1764: "thin near-bottom layer, zero total current").  The removed
    part Σ_thin w u is spread over the other layers in proportion to w, so the depth mean
    Σ w u — and with it the free-surface flux — is unchanged.  Faces whose layers are all thin
    are left alone.
    """
    thin = np.cumprod(hf < h_thin, axis=0).astype(bool)
    keep = ~thin
    w_keep = np.where(keep, w, 0.0).sum(axis=0)
    ok = w_keep > 0
    removed = np.where(thin, w * u, 0.0).sum(axis=0)
    shift = np.where(ok, removed / np.where(ok, w_keep, 1.0), 0.0)
    return np.where(thin & ok[None], 0.0, u + np.where(keep, shift[None], 0.0))


def lagrangian_thickness(h: np.ndarray, divq: np.ndarray, dt: float, area: np.ndarray,
                         tol: float = 1e-9) -> np.ndarray:
    """
    Layer continuity without flux through the interfaces (ω ≡ 0, spec §2 step 1):

        h_k† = h_k − Δt (∇·Q_k) / A

    divq (N, ny, nx) [m³ s⁻¹], area (ny, nx) [m²].  Values in [−tol, 0) are round-off and set
    to 0; a layer emptied beyond that means Δt is too long for the outflow (CFL) and raises.
    """
    out = h - dt * divq / np.where(area > 0, area, 1.0)[None]
    bad = out < -tol
    if bad.any():
        k, j, i = (int(a[0]) for a in np.nonzero(bad))
        raise RuntimeError(f"layer {k} of cell (j={j}, i={i}) emptied in the Lagrangian step: "
                           f"h† = {out[k, j, i]:.3e} m (reduce dt)")
    return np.maximum(out, 0.0)
