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
* PPM (Plan B2b, ``ppm_transport_x``): MOM6 continuity_PPM — positive-definite PPM edge
  thicknesses, the mean thickness of the volume swept in Δt, and a uniform velocity correction per
  column so the layer fluxes add up to the free-surface flux (Adcroft et al. 2019, §A).
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

import numba
import numpy as np

from core.vertical import remap as rm

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


def face_transport_thickness_numpy(h: np.ndarray, H: np.ndarray, active: np.ndarray,
                                   clip: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    NumPy reference of face_transport_thickness (test oracle).

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


@numba.njit(parallel=True, cache=True)
def _ftt_kernel(h, H, active, clip, q_e, q_w):  # pragma: no cover
    """Kernel of face_transport_thickness: one face column per (j, f), same operations as the NumPy version
    (NaN thickness or depth read as 0 here, as nan_to_num did; h and H may be strided views)."""
    n, ny, nx = h.shape
    for j in numba.prange(ny):
        for f in range(nx + 1):
            iw = f - 1 if f > 0 else 0
            ie = f if f < nx else nx - 1
            if not (active[j, iw] and active[j, ie]):
                for k in range(n):
                    q_e[k, j, f] = 0.0
                    q_w[k, j, f] = 0.0
                continue
            if clip:
                Hw = H[j, iw] if H[j, iw] == H[j, iw] else 0.0
                He = H[j, ie] if H[j, ie] == H[j, ie] else 0.0
                z_bed = max(-Hw, -He)
                cw = 0.0
                ce = 0.0
                for k in range(n):
                    hw = h[k, j, iw] if h[k, j, iw] == h[k, j, iw] else 0.0
                    he = h[k, j, ie] if h[k, j, ie] == h[k, j, ie] else 0.0
                    cw += hw
                    ce += he
                    q_e[k, j, f] = min(hw, max(0.0, (-Hw + cw) - z_bed))
                    q_w[k, j, f] = min(he, max(0.0, (-He + ce) - z_bed))
            else:
                for k in range(n):
                    hw = h[k, j, iw]
                    he = h[k, j, ie]
                    q_e[k, j, f] = hw if hw == hw else 0.0
                    q_w[k, j, f] = he if he == he else 0.0


def face_transport_thickness(h: np.ndarray, H: np.ndarray, active: np.ndarray,
                             clip: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Layer thickness available for transport through the u-faces [m]: q_E (eastward, from the
    west cell) and q_W (westward, from the east cell), each (N, ny, nx+1).  h (N, ny, nx) bed
    first, H (ny, nx) rest depth [m], active (ny, nx).  With ``clip`` (z*, hybrid) only the part of
    the upwind layer above the shallower bed passes (HYCOM cnuity.F90:252-256):

        q = min(h_k,up, max(0, z_top,k,up − max(−H_W, −H_E)))

    Without it (σ: the layers follow the terrain) q = h_k,up.  Domain-edge faces use the inner
    cell; faces next to land are 0.  For the v-faces call it on swapped axes.  Numba kernel
    (_ftt_kernel); face_transport_thickness_numpy is the reference.
    """
    h = np.asarray(h, float)
    n, ny, nx = h.shape
    q_e, q_w = np.empty((n, ny, nx + 1)), np.empty((n, ny, nx + 1))
    _ftt_kernel(h, np.asarray(H, float), np.asarray(active, np.bool_), bool(clip), q_e, q_w)
    return q_e, q_w


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


def face_fractions_numpy(hf: np.ndarray) -> np.ndarray:
    """NumPy reference of face_fractions (test oracle).  w_k = h_k,f / Σ_k h_k,f (uniform 1/N on faces with no water, so ū stays defined)."""
    tot = hf.sum(axis=0)
    wet = tot > 0
    return np.where(wet[None], hf / np.where(wet, tot, 1.0)[None], 1.0 / hf.shape[0])


def zero_thin_bed_velocity_numpy(u: np.ndarray, hf: np.ndarray, w: np.ndarray, h_thin: float = H_THIN_BED) -> np.ndarray:
    """NumPy reference of zero_thin_bed_velocity (test oracle).  
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


H_PPM_MIN = 2e-10      # [m] positive-definite floor of the PPM thickness (2 Angstrom, MOM_continuity_PPM.F90:534)


def ppm_edges_x(h: np.ndarray, active: np.ndarray, h_min: float = H_PPM_MIN) -> tuple[np.ndarray, np.ndarray]:
    """
    West and east edge values of the layer thickness in each cell [m] (MOM6 PPM_reconstruction_x,
    MOM_continuity_PPM.F90:2827-2942): limited slope (Lin 1994, eq. B2; 0 next to land),
    h_W = (h_i−1 + h_i)/2 + (s_i−1 − s_i)/6, a land or outside neighbour replaced by the cell
    itself, then the positive-definite limiter PPM_limit_pos (:3184-3226): a parabola whose
    minimum falls inside the cell is scaled so it stays ≥ h_min; cells with h ≤ h_min are flat.
    """
    h = np.where(active[None], np.nan_to_num(h), 0.0)
    m = active.astype(float)[None]
    z = np.zeros_like(m[..., :1])
    mW = np.concatenate([z, m[..., :-1]], axis=-1)
    mE = np.concatenate([m[..., 1:], z], axis=-1)
    hm1 = mW * np.concatenate([h[..., :1], h[..., :-1]], axis=-1) + (1 - mW) * h
    hp1 = mE * np.concatenate([h[..., 1:], h[..., -1:]], axis=-1) + (1 - mE) * h
    slp = 0.5 * (hp1 - hm1)
    dmx = np.maximum(np.maximum(hp1, hm1), h) - h
    dmn = h - np.minimum(np.minimum(hp1, hm1), h)
    slp = np.sign(slp) * np.minimum(np.abs(slp), 2.0 * np.minimum(dmx, dmn))
    slp = np.where(mW * mE * m > 0, slp, 0.0)
    zs = np.zeros_like(slp[..., :1])
    slp_w = np.concatenate([zs, slp[..., :-1]], axis=-1)
    slp_e = np.concatenate([slp[..., 1:], zs], axis=-1)
    h_w = 0.5 * (hm1 + h) + (slp_w - slp) / 6.0
    h_e = 0.5 * (hp1 + h) + (slp - slp_e) / 6.0
    curv = 3.0 * ((h_w + h_e) - 2.0 * h)                             # PPM_limit_pos: only minima
    dh = h_e - h_w
    inside = (curv > 0) & (np.abs(dh) < curv)
    flat = inside & (h <= h_min)
    num = 12.0 * curv * (h - h_min)
    den = curv**2 + 3.0 * dh**2
    scale_ = inside & ~flat & (num < den)
    s = np.where(scale_, num / np.where(den > 0, den, 1.0), 1.0)
    h_w = np.where(flat, h, h + s * (h_w - h))
    h_e = np.where(flat, h, h + s * (h_e - h))
    return h_w, h_e


def _swept_thickness(h, h_w, h_e, u, dt, dx_t, q_e, q_w):
    """
    Mean thickness of the volume swept through each u-face in dt and its derivative factor
    (MOM6 flux_elem, MOM_continuity_PPM.F90:1182-1183, 1193-1194), with C = |u| dt/dx of the
    donor capped at 1 and, where the bed clip is active (q < h_donor), capped by q.
    Returns (mean, marginal) on the faces (N, ny, nx+1).
    """
    pad = lambda a: np.concatenate([a[..., :1], a, a[..., -1:]], axis=-1)   # noqa: E731
    hp, wp, ep = pad(h), pad(h_w), pad(h_e)
    dxp = np.concatenate([dx_t[:, :1], dx_t, dx_t[:, -1:]], axis=-1)[None]
    # face f: west cell = padded index f, east cell = padded index f+1
    hW, wW, eW, dW = hp[..., :-1], wp[..., :-1], ep[..., :-1], dxp[..., :-1]
    hE, wE, eE, dE = hp[..., 1:], wp[..., 1:], ep[..., 1:], dxp[..., 1:]
    pos = u >= 0
    C = np.minimum(np.abs(u) * dt / np.where(pos, dW, dE), 1.0)
    curv3 = np.where(pos, (wW + eW) - 2.0 * hW, (wE + eE) - 2.0 * hE)
    dh = np.where(pos, wW - eW, eE - wE)
    edge = np.where(pos, eW, wE)
    mean = edge + C * (0.5 * dh + curv3 * (C - 1.5))
    marg = edge + C * (dh + 3.0 * curv3 * (C - 1.0))
    q = np.where(pos, q_e, q_w)
    h_up = np.where(pos, hW, hE)
    cap = np.where(q < h_up, q, np.inf)                              # bed clip (z*, hybrid)
    return np.maximum(np.minimum(mean, cap), 0.0), np.maximum(np.minimum(marg, cap), 0.0)


def _open_faces(active: np.ndarray) -> np.ndarray:
    """u-faces with water on both sides (domain-edge faces: the inner cell), (1, ny, nx+1)."""
    aW = np.concatenate([active[:, :1], active], axis=-1)
    aE = np.concatenate([active, active[:, -1:]], axis=-1)
    return (aW & aE)[None]


def _ppm_fluxes(h, h_w, h_e, u, L, q_e, q_w, open_, dt, dx_t):
    mean, marg = _swept_thickness(h, h_w, h_e, u, dt, dx_t, q_e, q_w)
    return np.where(open_, L[None] * u * mean, 0.0), np.where(open_, L[None] * marg, 0.0)


def ppm_fluxes_x(h: np.ndarray, ut: np.ndarray, L: np.ndarray, q_e: np.ndarray, q_w: np.ndarray,
                 active: np.ndarray, dt: float, dx_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Uncorrected PPM layer fluxes F = L u h̄ [m³ s⁻¹] and dF/du = L h_marg [m²] at the u-faces
    (MOM6 flux_elem, MOM_continuity_PPM.F90:1111-1202), zero on faces without water on both sides.
    """
    h = np.where(active[None], np.nan_to_num(h), 0.0)
    h_w, h_e = ppm_edges_x(h, active)
    return _ppm_fluxes(h, h_w, h_e, ut, L, q_e, q_w, _open_faces(active), dt, dx_t)


def ppm_transport_x_numpy(h: np.ndarray, ut: np.ndarray, L: np.ndarray, Q_bt: np.ndarray, q_e: np.ndarray,
                          q_w: np.ndarray, active: np.ndarray, dt: float, dx_t: np.ndarray,
                          max_iter: int = 20) -> np.ndarray:
    """
    NumPy reference of ppm_transport_x (test oracle).

    Layer volume fluxes through the u-faces [m³ s⁻¹] with PPM thickness (MOM6 continuity_PPM,
    OM4 defaults) and a uniform velocity correction du per face column so that
    Σ_k L (u_k + du) h̄_k(u_k + du) = Q_bt (zonal_flux_adjust, MOM_continuity_PPM.F90:1368-1546:
    Newton with bracketing bisection; du keeps every layer under CFL ½, :773-874).  The
    remaining residual is shared like HYCOM (cnuity.F90:585-620) so Σ_k Q_k = Q_bt to
    round-off.  Only the fluxes change, not u (as MOM6).  For the v-faces call it on swapped axes.
    """
    h = np.where(active[None], np.nan_to_num(h), 0.0)
    h_w, h_e = ppm_edges_x(h, active)
    open_ = _open_faces(active)
    dxW = np.concatenate([dx_t[:, :1], dx_t], axis=-1)[None]
    dxE = np.concatenate([dx_t, dx_t[:, -1:]], axis=-1)[None]

    def fluxes(uu):
        return _ppm_fluxes(h, h_w, h_e, uu, L, q_e, q_w, open_, dt, dx_t)

    lim_w, lim_e = 0.5 * dxW / dt, 0.5 * dxE / dt                    # CFL ½ (CONTINUITY_CFL_LIMIT)
    du_max = np.maximum(np.minimum(lim_w[0], (lim_w - ut).min(axis=0)), 0.0)
    du_min = np.minimum(np.maximum(-lim_e[0], (-(lim_e + ut)).max(axis=0)), 0.0)
    du = np.zeros_like(Q_bt)
    F, dF = fluxes(ut)
    err = F.sum(axis=0) - Q_bt
    lo, hi = du_min.copy(), du_max.copy()
    for _ in range(max_iter):
        hi = np.where(err > 0, du, hi)
        lo = np.where(err < 0, du, lo)
        dsum = dF.sum(axis=0)
        step = np.where(dsum > 0, -err / np.where(dsum > 0, dsum, 1.0), 0.0)
        new = du + step
        new = np.where((step > 0) & (new >= hi), 0.5 * (du + hi), new)
        new = np.where((step < 0) & (new <= lo), 0.5 * (du + lo), new)
        du = new
        F, dF = fluxes(ut + du[None])
        err = F.sum(axis=0) - Q_bt
        if np.all(np.abs(err) <= 1e-12 * np.maximum(np.abs(Q_bt), 1.0)):
            break
    R = Q_bt - F.sum(axis=0)                                         # exact closure (HYCOM)
    qR = np.where((R >= 0)[None], q_e, q_w)
    alt = np.where((R >= 0)[None], q_w, q_e)
    qR = np.where((qR.sum(axis=0) > 0)[None], qR, alt)
    tot = qR.sum(axis=0)
    return np.where(open_, F + R[None] * qR / np.where(tot > 0, tot, 1.0)[None], 0.0)


@numba.njit(parallel=True, cache=True)
def _ppm_edges_kernel(h, active, h_min, h_w, h_e):  # pragma: no cover
    """Kernel of ppm_edges_x (same operations, MOM_continuity_PPM.F90:2827-2942 and 3184-3226)."""
    n, ny, nx = h.shape
    for j in numba.prange(ny):
        slp = np.empty(nx)
        for k in range(n):
            for i in range(nx):
                m = 1.0 if active[j, i] else 0.0
                mw = (1.0 if active[j, i - 1] else 0.0) if i > 0 else 0.0
                me = (1.0 if active[j, i + 1] else 0.0) if i < nx - 1 else 0.0
                hc = h[k, j, i]
                hm1 = mw * h[k, j, i - 1] + (1.0 - mw) * hc if i > 0 else hc
                hp1 = me * h[k, j, i + 1] + (1.0 - me) * hc if i < nx - 1 else hc
                s = 0.5 * (hp1 - hm1)
                dmx = max(max(hp1, hm1), hc) - hc
                dmn = hc - min(min(hp1, hm1), hc)
                a = min(abs(s), 2.0 * min(dmx, dmn))
                s = a if s > 0.0 else (-a if s < 0.0 else 0.0)
                slp[i] = s if mw * me * m > 0.0 else 0.0
            for i in range(nx):
                mw = (1.0 if active[j, i - 1] else 0.0) if i > 0 else 0.0
                me = (1.0 if active[j, i + 1] else 0.0) if i < nx - 1 else 0.0
                hc = h[k, j, i]
                hm1 = mw * h[k, j, i - 1] + (1.0 - mw) * hc if i > 0 else hc
                hp1 = me * h[k, j, i + 1] + (1.0 - me) * hc if i < nx - 1 else hc
                sw = slp[i - 1] if i > 0 else 0.0
                se = slp[i + 1] if i < nx - 1 else 0.0
                hw = 0.5 * (hm1 + hc) + (sw - slp[i]) / 6.0
                he = 0.5 * (hp1 + hc) + (slp[i] - se) / 6.0
                curv = 3.0 * ((hw + he) - 2.0 * hc)
                dh = he - hw
                inside = curv > 0.0 and abs(dh) < curv
                if inside and hc <= h_min:
                    h_w[k, j, i] = hc
                    h_e[k, j, i] = hc
                    continue
                sc = 1.0
                if inside:
                    num = 12.0 * curv * (hc - h_min)
                    den = curv * curv + 3.0 * dh * dh
                    if num < den:
                        sc = num / den
                h_w[k, j, i] = hc + sc * (hw - hc)
                h_e[k, j, i] = hc + sc * (he - hc)


@numba.njit(cache=True)
def _face_flux(k, j, f, u, h, h_w, h_e, dx_t, q_e, q_w, L_f, dt, nx):  # pragma: no cover
    """(F, dF/du) of layer k at u-face (j, f): _swept_thickness + _ppm_fluxes for one open face (MOM6 flux_elem)."""
    iw = f - 1 if f > 0 else 0
    ie = f if f < nx else nx - 1
    if u >= 0.0:
        c = min(abs(u) * dt / dx_t[j, iw], 1.0)
        hu, ww, ee = h[k, j, iw], h_w[k, j, iw], h_e[k, j, iw]
        curv3, dh, edge, q = (ww + ee) - 2.0 * hu, ww - ee, ee, q_e[k, j, f]
    else:
        c = min(abs(u) * dt / dx_t[j, ie], 1.0)
        hu, ww, ee = h[k, j, ie], h_w[k, j, ie], h_e[k, j, ie]
        curv3, dh, edge, q = (ww + ee) - 2.0 * hu, ee - ww, ww, q_w[k, j, f]
    mean = edge + c * (0.5 * dh + curv3 * (c - 1.5))
    marg = edge + c * (dh + 3.0 * curv3 * (c - 1.0))
    if q < hu:                                                         # bed clip (z*, hybrid)
        mean = min(mean, q)
        marg = min(marg, q)
    return L_f * u * max(mean, 0.0), L_f * max(marg, 0.0)


@numba.njit(parallel=True, cache=True)
def _ppm_transport_kernel(h, h_w, h_e, ut, L, Q_bt, q_e, q_w, active, dt, dx_t, max_iter, out):  # pragma: no cover
    """
    Kernel of ppm_transport_x_numpy: Newton with bracketing bisection on the uniform correction du of each face
    column (zonal_flux_adjust, MOM_continuity_PPM.F90:1368-1546), stopped when *all* faces have converged, as
    the NumPy loop (identical iteration count), then the HYCOM residual share (cnuity.F90:585-620).
    """
    n, ny, nx = h.shape
    nf = nx + 1
    du = np.zeros((ny, nf))
    lo = np.empty((ny, nf))
    hi = np.empty((ny, nf))
    err = np.empty((ny, nf))
    dsum = np.empty((ny, nf))
    for j in numba.prange(ny):
        for f in range(nf):
            iw = f - 1 if f > 0 else 0
            ie = f if f < nx else nx - 1
            limw, lime = 0.5 * dx_t[j, iw] / dt, 0.5 * dx_t[j, ie] / dt
            mnw, mxe = np.inf, -np.inf
            for k in range(n):
                mnw = min(mnw, limw - ut[k, j, f])
                mxe = max(mxe, -(lime + ut[k, j, f]))
            hi[j, f] = max(min(limw, mnw), 0.0)
            lo[j, f] = min(max(-lime, mxe), 0.0)
            s, sd = 0.0, 0.0
            if active[j, iw] and active[j, ie]:
                for k in range(n):
                    F, dF = _face_flux(k, j, f, ut[k, j, f], h, h_w, h_e, dx_t, q_e, q_w, L[j, f], dt, nx)
                    s += F
                    sd += dF
            err[j, f] = s - Q_bt[j, f]
            dsum[j, f] = sd
    for _ in range(max_iter):
        for j in numba.prange(ny):
            for f in range(nf):
                e, d = err[j, f], du[j, f]
                if e > 0.0:
                    hi[j, f] = d
                if e < 0.0:
                    lo[j, f] = d
                step = -e / dsum[j, f] if dsum[j, f] > 0.0 else 0.0
                new = d + step
                if step > 0.0 and new >= hi[j, f]:
                    new = 0.5 * (d + hi[j, f])
                if step < 0.0 and new <= lo[j, f]:
                    new = 0.5 * (d + lo[j, f])
                du[j, f] = new
                iw = f - 1 if f > 0 else 0
                ie = f if f < nx else nx - 1
                s, sd = 0.0, 0.0
                if active[j, iw] and active[j, ie]:
                    for k in range(n):
                        F, dF = _face_flux(k, j, f, ut[k, j, f] + new, h, h_w, h_e, dx_t, q_e, q_w, L[j, f], dt, nx)
                        s += F
                        sd += dF
                err[j, f] = s - Q_bt[j, f]
                dsum[j, f] = sd
        done = True
        for j in range(ny):
            for f in range(nf):
                if abs(err[j, f]) > 1e-12 * max(abs(Q_bt[j, f]), 1.0):
                    done = False
        if done:
            break
    for j in numba.prange(ny):
        Fk = np.empty(n)
        for f in range(nf):
            iw = f - 1 if f > 0 else 0
            ie = f if f < nx else nx - 1
            if not (active[j, iw] and active[j, ie]):
                for k in range(n):
                    out[k, j, f] = 0.0
                continue
            s = 0.0
            for k in range(n):
                Fk[k] = _face_flux(k, j, f, ut[k, j, f] + du[j, f], h, h_w, h_e, dx_t, q_e, q_w, L[j, f], dt, nx)[0]
                s += Fk[k]
            R = Q_bt[j, f] - s
            te, tw = 0.0, 0.0
            for k in range(n):
                te += q_e[k, j, f]
                tw += q_w[k, j, f]
            use_e = R >= 0.0
            tot = te if use_e else tw
            if tot <= 0.0:
                use_e = not use_e
                tot = te if use_e else tw
            den = tot if tot > 0.0 else 1.0
            for k in range(n):
                out[k, j, f] = Fk[k] + R * (q_e[k, j, f] if use_e else q_w[k, j, f]) / den


def ppm_transport_x(h: np.ndarray, ut: np.ndarray, L: np.ndarray, Q_bt: np.ndarray, q_e: np.ndarray,
                    q_w: np.ndarray, active: np.ndarray, dt: float, dx_t: np.ndarray,
                    max_iter: int = 20) -> np.ndarray:
    """
    Layer volume fluxes through the u-faces [m³ s⁻¹] with PPM thickness (MOM6 continuity_PPM,
    OM4 defaults) and a uniform velocity correction du per face column so that
    Σ_k L (u_k + du) h̄_k(u_k + du) = Q_bt (zonal_flux_adjust, MOM_continuity_PPM.F90:1368-1546:
    Newton with bracketing bisection; du keeps every layer under CFL ½, :773-874).  The
    remaining residual is shared like HYCOM (cnuity.F90:585-620) so Σ_k Q_k = Q_bt to
    round-off.  Only the fluxes change, not u (as MOM6).  For the v-faces call it on swapped axes.
    Numba kernels (_ppm_edges_kernel, _ppm_transport_kernel); ppm_transport_x_numpy is the reference.
    """
    hc = np.ascontiguousarray(np.where(active[None], np.nan_to_num(h), 0.0))
    act = np.ascontiguousarray(active)
    h_w, h_e = np.empty_like(hc), np.empty_like(hc)
    _ppm_edges_kernel(hc, act, H_PPM_MIN, h_w, h_e)
    out = np.empty(np.shape(ut))
    c = np.asarray                                                     # views pass as they are (perf part 2)
    _ppm_transport_kernel(hc, h_w, h_e, c(ut, dtype=float), c(L, dtype=float), c(Q_bt, dtype=float),
                          c(q_e, dtype=float), c(q_w, dtype=float), act, float(dt), c(dx_t, dtype=float),
                          int(max_iter), out)
    return out


def transport_velocity_numpy(u_new: np.ndarray, u_old: np.ndarray, last_wet: np.ndarray, w: np.ndarray,
                             L: np.ndarray, D: np.ndarray, theta: float,
                             theta_bc: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    NumPy reference of transport_velocity (test oracle).  Transport velocity of the ALE step: the depth mean
    ū = Σ_k w_k u_k with the free-surface θ (its volume flux must be the free-surface flux), the baroclinic part
    u′ = u − ū with θ_bc (time-centred, θ_bc = ½: the AB3 momentum tendency is centred at n + ½; a θ_bc > ½ damps
    internal waves at ~(θ_bc − ½) ω² Δt — Ilicak internal wave, validation/ilicak_benchmark):

        u^θ_k = θ_bc u_k^{n+1} + (1 − θ_bc) u_k^n + (θ − θ_bc)(ū^{n+1} − ū^n)

    θ_bc = None means θ (the scheme before 2026-10-13).  u^n = 0 on faces that carried no flow in the
    free-surface step (``last_wet`` False); Q_bt = L D Σ_k w_k u^θ_k [m³ s⁻¹] (Σ_k w_k = 1, face_fractions).
    u (N, a, b), last_wet, L, D (a, b), w (N, a, b) layer fractions.
    """
    tb = theta if theta_bc is None else theta_bc
    un = np.where(last_wet[None], u_old, 0.0)
    m1, m0 = (w * u_new).sum(axis=0), (w * un).sum(axis=0)
    ut = tb * u_new + (1 - tb) * un + ((theta - tb) * (m1 - m0))[None]
    return ut, L * D * (w * ut).sum(axis=0)


@numba.njit(parallel=True, cache=True)
def _transport_velocity_kernel(u_new, u_old, last_wet, w, L, D, theta, theta_bc, ut, Qb):  # pragma: no cover
    """transport_velocity_numpy with the same arithmetic, one face column per thread item."""
    n, a, b = u_new.shape
    for p in numba.prange(a * b):
        j, i = p // b, p % b
        m1 = 0.0
        m0 = 0.0
        for k in range(n):
            un = u_old[k, j, i] if last_wet[j, i] else 0.0
            m1 += w[k, j, i] * u_new[k, j, i]
            m0 += w[k, j, i] * un
        d = (theta - theta_bc) * (m1 - m0)
        s = 0.0
        for k in range(n):
            un = u_old[k, j, i] if last_wet[j, i] else 0.0
            x = theta_bc * u_new[k, j, i] + (1 - theta_bc) * un + d
            ut[k, j, i] = x
            s += w[k, j, i] * x
        Qb[j, i] = L[j, i] * D[j, i] * s


def transport_velocity(u_new: np.ndarray, u_old: np.ndarray, last_wet: np.ndarray, w: np.ndarray,
                       L: np.ndarray, D: np.ndarray, theta: float,
                       theta_bc: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(u^θ, Q_bt) of the ALE step (see transport_velocity_numpy, the reference).  Numba kernel."""
    n, a, b = np.shape(u_new)
    if np.shape(u_old) != (n, a, b) or np.shape(w) != (n, a, b) or any(
            np.shape(x) != (a, b) for x in (last_wet, L, D)):         # the kernel does not check bounds
        raise ValueError(f"transport_velocity: shapes u_new {np.shape(u_new)}, u_old {np.shape(u_old)}, "
                         f"w {np.shape(w)}, last_wet/L/D {np.shape(last_wet)}/{np.shape(L)}/{np.shape(D)}")
    ut, Qb = np.empty((n, a, b)), np.empty((a, b))
    _transport_velocity_kernel(np.asarray(u_new, float), np.asarray(u_old, float), np.asarray(last_wet, np.bool_),
                               np.asarray(w, float), np.asarray(L, float), np.asarray(D, float), float(theta),
                               float(theta if theta_bc is None else theta_bc), ut, Qb)
    return ut, Qb


# ---------------------------------------------------------------- open boundaries (Plan C, scope A)
def baroclinic_split_numpy(vel: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy reference of baroclinic_split (test oracle).  
    Depth mean and baroclinic part of a face velocity with layer weights w (Σ_k w_k = 1):

        ū = Σ_k w_k u_k,   u′_k = u_k − ū,   so Σ_k w_k u′_k = 0

    w = Δσ in σ, the face fractions ``face_fractions`` in ALE (spec 2026-10-12-ale-open-boundaries §3.1).
    vel, w (N, …); returns (ū (…), u′ (N, …)) [m s⁻¹].
    """
    mean = (w * vel).sum(axis=0)
    return mean, vel - mean[None]


def open_face_transport(h_int: np.ndarray, u_new: np.ndarray, u_old: np.ndarray, Q_bt: np.ndarray,
                        L: np.ndarray, theta: float, theta_bc: float | None = None) -> np.ndarray:
    """
    Layer volume fluxes through the faces of one open edge [m³ s⁻¹] (spec §3.4):

        Q_k = (h_k / Σ_j h_j) Q_bt + L h_k (θ_bc u′_kⁿ⁺¹ + (1 − θ_bc) u′_kⁿ)

    (θ_bc: the time weight of the baroclinic part, as transport_velocity; None means θ)

    with h_k the thickness of the interior cell (MOM6 MOM_continuity_PPM.F90:1237-1248) and u′ split
    with the interior fractions h_k/Σh, so Σ_k Q_k = Q_bt (the Flather flux of the free surface) to
    round-off and a vanished layer carries nothing.  h_int, u_new, u_old (N, m); Q_bt, L (m,).
    """
    tot = h_int.sum(axis=0)
    wet = tot > 0
    w = np.where(wet[None], h_int / np.where(wet, tot, 1.0)[None], 0.0)
    pn = u_new - (w * u_new).sum(axis=0)[None]
    po = u_old - (w * u_old).sum(axis=0)[None]
    tb = theta if theta_bc is None else theta_bc
    Q = w * Q_bt[None] + L[None] * h_int * (tb * pn + (1.0 - tb) * po)
    return np.where(wet[None], Q, 0.0)


def exterior_on_layers(h_ext: np.ndarray, T_ext: np.ndarray, h: np.ndarray, mask: np.ndarray,
                       scheme: int) -> np.ndarray:
    """
    Exterior tracer profile on the current layers (spec §3.5): the profile (h_ext, T_ext), kept in z
    since the initial state, is stretched to the current column, h_ext Σh/Σh_ext (= (H + η)/H for a
    resting initial state, z*-like), and remapped conservatively (core.vertical.remap, ``scheme``)
    onto the thicknesses h — the external data on its own z axis of MOM6 (MOM_open_boundary.F90:1188-1204),
    on the *current* thicknesses (avoiding OBC_TRACER_DZ_BUG, :778-781).  Columns outside ``mask``
    (ny, nx) or without water keep T_ext.  (N, ny, nx) arrays.
    """
    tot_e, tot = h_ext.sum(axis=0), h.sum(axis=0)
    wet = np.asarray(mask, bool) & (tot_e > 0) & (tot > 0)
    scale = np.where(wet, tot / np.where(wet, tot_e, 1.0), 1.0)
    out = rm.remap_columns(h_ext * scale[None], T_ext, h, scheme, wet)
    return np.where(wet[None], out, T_ext)


# ---------------------------------------------------------------- Numba kernels (ALE performance task 5.4)
@numba.njit(parallel=True, cache=True)
def _fractions_kernel(hf, w):  # pragma: no cover
    """face_fractions_numpy with the same arithmetic (sequential sum over the layers)."""
    n, a, b = hf.shape
    for p in numba.prange(a * b):
        j, i = p // b, p % b
        tot = 0.0
        for k in range(n):
            tot += hf[k, j, i]
        for k in range(n):
            w[k, j, i] = hf[k, j, i] / tot if tot > 0.0 else 1.0 / n


def face_fractions(hf: np.ndarray) -> np.ndarray:
    """w_k = h_k,f / Σ_k h_k,f (uniform 1/N on faces with no water, so ū stays defined).  Numba kernel;
    face_fractions_numpy is the reference."""
    if np.ndim(hf) != 3:
        return face_fractions_numpy(hf)
    w = np.empty(np.shape(hf))
    _fractions_kernel(np.asarray(hf, dtype=float), w)
    return w


@numba.njit(parallel=True, cache=True)
def _split_kernel(vel, w, mean, prime):  # pragma: no cover
    """baroclinic_split_numpy with the same arithmetic."""
    n, a, b = vel.shape
    for p in numba.prange(a * b):
        j, i = p // b, p % b
        m = 0.0
        for k in range(n):
            m += w[k, j, i] * vel[k, j, i]
        mean[j, i] = m
        for k in range(n):
            prime[k, j, i] = vel[k, j, i] - m


def baroclinic_split(vel: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Depth mean and baroclinic part of a face velocity with layer weights w (Σ_k w_k = 1):
    ū = Σ_k w_k u_k, u′_k = u_k − ū (spec 2026-10-12-ale-open-boundaries §3.1).  Numba kernel for full
    (N, a, b) arrays; baroclinic_split_numpy is the reference (and handles broadcast weights).
    """
    if np.ndim(vel) != 3 or np.shape(w) != np.shape(vel):
        return baroclinic_split_numpy(vel, w)
    mean, prime = np.empty(np.shape(vel)[1:]), np.empty(np.shape(vel))
    _split_kernel(np.asarray(vel, dtype=float), np.asarray(w, dtype=float), mean, prime)
    return mean, prime


@numba.njit(parallel=True, cache=True)
def _thin_bed_kernel(u, hf, w, h_thin, out):  # pragma: no cover
    """zero_thin_bed_velocity_numpy with the same arithmetic (thin run from the bed, sequential sums)."""
    n, a, b = u.shape
    for p in numba.prange(a * b):
        j, i = p // b, p % b
        m = 0
        while m < n and hf[m, j, i] < h_thin:
            m += 1
        w_keep = 0.0
        removed = 0.0
        for k in range(n):
            if k >= m:
                w_keep += w[k, j, i]
            else:
                removed += w[k, j, i] * u[k, j, i]
        ok = w_keep > 0.0
        shift = removed / w_keep if ok else 0.0
        for k in range(n):
            if k < m:
                out[k, j, i] = 0.0 if ok else u[k, j, i] + 0.0
            else:
                out[k, j, i] = u[k, j, i] + shift


def zero_thin_bed_velocity(u: np.ndarray, hf: np.ndarray, w: np.ndarray, h_thin: float = H_THIN_BED) -> np.ndarray:
    """
    Zero velocity in the run of face layers thinner than ``h_thin`` that starts at the bed
    (HYCOM hybgen.F90:1757-1764: "thin near-bottom layer, zero total current"); the removed part
    Σ_thin w u is spread over the other layers in proportion to w, so Σ w u is unchanged.  Faces whose
    layers are all thin are left alone.  Numba kernel; zero_thin_bed_velocity_numpy is the reference.
    """
    if np.ndim(u) != 3 or np.shape(hf) != np.shape(u) or np.shape(w) != np.shape(u):
        return zero_thin_bed_velocity_numpy(u, hf, w, h_thin)
    out = np.empty(np.shape(u))
    _thin_bed_kernel(np.asarray(u, dtype=float), np.asarray(hf, dtype=float),
                     np.asarray(w, dtype=float), float(h_thin), out)
    return out
