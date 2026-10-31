"""
Metrics of the DOME benchmark (spec 2026-10-14-dome-overflow §7; Legg et al. 2006 eq. 11–14, 18, Figs. 4, 15), the same
functions for every model on the common series (validation/dome/series.py; layers surface first, h ≥ 0, land NaN).
x is along the slope (the plume turns west: transports westward positive), y across it.
"""
from __future__ import annotations

import warnings

import numpy as np

from diagnostics.mixing import reference_potential_energy
from validation.dome import cases
from validation.ilicak_benchmark.metrics import hypsometry


def _window(s: dict, t0: float, t1: float) -> np.ndarray:
    """Indices of the records with t0 ≤ t ≤ t1 [s]."""
    return np.nonzero((s["t"] >= t0 - 1e-6) & (s["t"] <= t1 + 1e-6))[0]


def _vol(s: dict) -> np.ndarray:
    """Cell volumes h Δx Δy [m³] of every record."""
    return np.nan_to_num(s["h"]) * (s["dy"][:, None] * s["dx"][None, :])[None, None]


def _rho(T: np.ndarray) -> np.ndarray:
    """Linear EOS of the case [kg m⁻³]."""
    return cases.RHO0 - cases.ALPHA * (T - cases.T_REF)


def path_Y(s: dict, t0: float, t1: float):
    """Y(x) = ∫ y s dy dz / ∫ s dy dz (Legg eq. 18), mean over the records in [t0, t1] [m]; NaN where no dye."""
    ys = []
    for n in _window(s, t0, t1):
        w = np.nan_to_num(s["s"][n]) * np.nan_to_num(s["h"][n]) * s["dy"][None, :, None]
        num = (w * s["y"][None, :, None]).sum(axis=(0, 1))
        den = w.sum(axis=(0, 1))
        ys.append(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan))
    with warnings.catch_warnings():                                      # columns without dye: NaN, no warning
        warnings.simplefilter("ignore", RuntimeWarning)
        return s["x"], np.nanmean(ys, axis=0)


def plume_transport(s: dict, t0: float, t1: float, thr: float = 0.01):
    """T(x) = ∫_A (−u) dy dz, A = {s > thr} (Legg eq. 11), at cell centres (u averaged to them), time mean [m³ s⁻¹]."""
    out = []
    for n in _window(s, t0, t1):
        uc = 0.5 * (s["u"][n][..., 1:] + s["u"][n][..., :-1])
        A = np.nan_to_num(s["s"][n]) > thr
        out.append((np.where(A, -uc * np.nan_to_num(s["h"][n]), 0.0) * s["dy"][None, :, None]).sum(axis=(0, 1)))
    return s["x"], np.mean(out, axis=0)


def entrainment(s: dict, t0: float, t1: float, thr: float = 0.01):
    """α_E(x) = (dT/dx)/(L Ū) (Legg eq. 12–13): L the length of the s = thr interface in (y, z) [m], Ū = T/∫_A dy dz."""
    x, T = plume_transport(s, t0, t1, thr)
    Ls, As = [], []
    for n in _window(s, t0, t1):
        A = np.nan_to_num(s["s"][n]) > thr
        h = np.nan_to_num(s["h"][n])
        As.append((A * h * s["dy"][None, :, None]).sum(axis=(0, 1)))
        edge_y = (A[:, 1:] != A[:, :-1]) * 0.5 * (h[:, 1:] + h[:, :-1])          # vertical pieces of the interface
        edge_z = (A[1:] != A[:-1]) * s["dy"][None, :, None]                      # horizontal pieces
        Ls.append(edge_y.sum(axis=(0, 1)) + edge_z.sum(axis=(0, 1)))
    L, area = np.mean(Ls, axis=0), np.mean(As, axis=0)
    with np.errstate(all="ignore"):
        U = np.where(area > 0, T / np.where(area > 0, area, 1.0), np.nan)
        dTdx = -np.gradient(T, x)                                         # westward growth (x decreasing)
        return x, np.where((L > 0) & (U > 0), dTdx / (L * U), np.nan)


X_INLET, Y_TOP = 1050e3, 600e3         # Legg's origin: x' = x − 1050 km (inlet centre), y' = y − 600 km (slope top)
CORNER = (-50e3, 50e3, -50e3, 0.0)      # Legg Fig. 14 caption: −50 < x' < 50 km, −50 < y' < 0 km


def _plume_depths(h_col: np.ndarray, a_col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth intervals of one column (layers surface first): interface depths z (nz+1,) [m, from the surface, down]
    and the plume indicator of each layer; layers of zero thickness drop out (they hold no water)."""
    z = np.concatenate([[0.0], np.cumsum(h_col)])
    keep = h_col > 0.0
    return z, a_col & keep


def _lateral_area(za: np.ndarray, aa: np.ndarray, zb: np.ndarray, ab: np.ndarray) -> float:
    """∫ |χ_a(z) − χ_b(z)| dz over the depths where both columns hold water [m]: the vertical piece of the
    s = thr surface between two neighbouring columns (exact for any layering: z, σ, ALE)."""
    d = min(za[-1], zb[-1])
    if d <= 0.0:
        return 0.0
    zz = np.unique(np.concatenate([za, zb, [d]]))
    zz = zz[zz <= d]
    mid = 0.5 * (zz[1:] + zz[:-1])
    ia = np.clip(np.searchsorted(za, mid) - 1, 0, aa.size - 1)
    ib = np.clip(np.searchsorted(zb, mid) - 1, 0, ab.size - 1)
    return float((np.diff(zz) * (aa[ia] != ab[ib])).sum())


def _top_transitions(z: np.ndarray, a: np.ndarray) -> int:
    """Number of plume/ambient changes down one column, between water-holding layers (the horizontal pieces)."""
    w = np.diff(z) > 0.0
    seq = a[w]
    return int((seq[1:] != seq[:-1]).sum())


def corner_entrainment(s: dict, t0: float, t1: float, thr: float = 0.01, urms: str = "plume",
                       parts: bool = False):
    """
    Mean entrainment coefficient of the corner region (Legg et al. 2006 eq. 14, Fig. 14):

        E = (⟨T_out⟩ − ⟨T_in⟩) / (⟨U_rms⟩ ⟨S⟩)

    with the region −50 < x' < 50 km, −50 < y' < 0 km (cells whose centres lie inside), ⟨·⟩ the mean over the
    records in [t0, t1] and A = {s > thr}:
    - T_out [m³ s⁻¹]: westward transport ∫_A (−u) dy dz through the face x' = −50 km, rows of the region;
    - T_in [m³ s⁻¹]: southward transport ∫_A (−v) dx dz through the face y' = 0, columns of the region
      (s, h at a face: the mean of its two cells);
    - S [m²]: area of the s = thr surface inside the region: the plume/ambient changes down each column (Δx Δy each)
      plus the vertical pieces between neighbouring region columns, ∫|χ_a − χ_b| dz over the depths where both
      hold water (faces against the bed or the region's edges do not count);
    - U_rms [m s⁻¹]: √(volume mean of u² + v²) at cell centres over the plume cells of the region (``urms='plume'``)
      or over all its water (``urms='all'``); Legg's text says only "the rms horizontal velocity scale within the
      corner region".
    Legg's non-hydrostatic Case 1 value is 8.6e-4 (p. 79).  ``parts`` True returns the dict of the four means and E.
    """
    x0, x1, y0, y1 = CORNER
    xp, yp = s["x"] - X_INLET, s["y"] - Y_TOP
    ii = np.nonzero((xp > x0) & (xp < x1))[0]
    jj = np.nonzero((yp > y0) & (yp < y1))[0]
    i_f = ii[0]                                         # u-face west of the first region column: x' = −50 km
    j_f = jj[-1] + 1                                    # v-face north of the last region row: y' = 0
    dx, dy = s["dx"], s["dy"]
    acc = dict(T_out=[], T_in=[], S=[], U_rms=[])
    for n in _window(s, t0, t1):
        h = np.nan_to_num(s["h"][n])
        sn = np.nan_to_num(s["s"][n])
        A = sn > thr
        # T_out through x' = −50 km
        sf = 0.5 * (sn[:, jj, i_f - 1] + sn[:, jj, i_f])
        hf = 0.5 * (h[:, jj, i_f - 1] + h[:, jj, i_f])
        acc["T_out"].append(float((np.where(sf > thr, -s["u"][n][:, jj, i_f] * hf, 0.0) * dy[None, jj]).sum()))
        # T_in through y' = 0
        sf = 0.5 * (sn[:, j_f - 1, ii] + sn[:, j_f, ii])
        hf = 0.5 * (h[:, j_f - 1, ii] + h[:, j_f, ii])
        acc["T_in"].append(float((np.where(sf > thr, -s["v"][n][:, j_f, ii] * hf, 0.0) * dx[None, ii]).sum()))
        # S: horizontal and vertical pieces inside the region
        col = {(j, i): _plume_depths(h[:, j, i], A[:, j, i]) for j in jj for i in ii}
        area = 0.0
        for (j, i), (z, a) in col.items():
            area += _top_transitions(z, a) * dx[i] * dy[j]
            if (j, i + 1) in col:
                area += _lateral_area(z, a, *col[(j, i + 1)]) * dy[j]
            if (j + 1, i) in col:
                area += _lateral_area(z, a, *col[(j + 1, i)]) * dx[i]
        acc["S"].append(area)
        # U_rms
        sub = (slice(None), jj[:, None], ii[None, :])
        uc = 0.5 * (s["u"][n][..., 1:] + s["u"][n][..., :-1])[sub]
        vc = 0.5 * (s["v"][n][:, 1:, :] + s["v"][n][:, :-1, :])[sub]
        vol = (h * (dy[:, None] * dx[None, :])[None])[sub]
        w = vol * (A[sub] if urms == "plume" else 1.0)
        acc["U_rms"].append(float(np.sqrt((w * (uc ** 2 + vc ** 2)).sum() / w.sum())) if w.sum() > 0 else np.nan)
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    m["E"] = (m["T_out"] - m["T_in"]) / (m["U_rms"] * m["S"]) if m["S"] > 0 else float("nan")
    return m if parts else m["E"]


def buoyancy(T: np.ndarray) -> np.ndarray:
    """b = −g (ρ − ρ_top)/ρ₀ [m s⁻²] (Legg's buoyancy, 0 at the surface, −0.019 for the densest water)."""
    return -cases.G * (_rho(T) - cases.RHO_TOP) / cases.RHO0


def tracer_by_class(s: dict, t0: float, t1: float, n_classes: int = 20, b_min: float = -0.02):
    """
    Total tracer against along-slope distance and buoyancy class (Legg et al. 2006 Fig. 4), time mean over [t0, t1]:
    S(c, x) = Σ_{cells of column x with b in class c} s h Δy  [m²]  (∫ s dy dz per x and class; Legg's colour scale,
    ×10⁶, is of this size).  Classes: n_classes equal bins of b in [b_min, 0] (default 20 × 0.001 m s⁻², Legg's
    bins), b outside folded into the end bins.  Returns (x [m], b_edges (n_classes+1,), S (n_classes, nx)).
    Σ_c Σ_x S Δx is the time mean of Σ s V (the dye content).
    """
    edges = np.linspace(b_min, 0.0, n_classes + 1)
    idx = _window(s, t0, t1)
    out = np.zeros((n_classes, s["x"].size))
    for n in idx:
        w = np.nan_to_num(s["s"][n]) * np.nan_to_num(s["h"][n]) * s["dy"][None, :, None]
        k = np.clip(np.searchsorted(edges, buoyancy(np.nan_to_num(s["T"][n])), side="right") - 1, 0, n_classes - 1)
        for c in range(n_classes):
            out[c] += np.where(k == c, w, 0.0).sum(axis=(0, 1))
    return s["x"], edges, out / max(1, idx.size)


def final_density(s: dict, t0: float, t1: float, x_max: float = 800e3) -> float:
    """Dye-weighted mean density of the plume west of x_max (Legg Fig. 15 'final buoyancy'), time mean [kg m⁻³]."""
    vals = []
    for n in _window(s, t0, t1):
        w = np.nan_to_num(s["s"][n]) * _vol(s)[n] * (s["x"] < x_max)[None, None, :]
        vals.append((w * _rho(np.nan_to_num(s["T"][n]))).sum() / w.sum())
    return float(np.mean(vals))


def centroid_depth(s: dict, t: float) -> float:
    """Dye-weighted mean depth of the layer centres at the record nearest t [m, positive down]."""
    n = int(np.argmin(np.abs(s["t"] - t)))
    h = np.nan_to_num(s["h"][n])
    zc = np.cumsum(h, axis=0) - 0.5 * h - np.nan_to_num(s["eta"][n])[None]
    w = np.nan_to_num(s["s"][n]) * _vol(s)[n]
    return float((w * zc).sum() / w.sum())


def ranges(s: dict) -> dict:
    """(min, max) of T and s over every wet cell of every record."""
    wet = np.nan_to_num(s["h"]) > 0
    return {k: (float(np.min(s[k][wet])), float(np.max(s[k][wet]))) for k in ("T", "s")}


def rpe_series(s: dict) -> np.ndarray:
    """RPE [J] of every record (Ilicak's exact hypsometry; validation/ilicak_benchmark/metrics.py)."""
    zA = hypsometry(s["H"], s["dx"], s["dy"], top=60.0)                   # the sea level rises ~35 m in 50 days
    vol = _vol(s)
    rho = _rho(s["T"])
    return np.array([reference_potential_energy(np.where(vol[n] > 0, rho[n], np.nan), vol[n], hypsometry=zA)
                     for n in range(s["t"].size)])


def inflow_by_class(s: dict, t0: float, t1: float, n_classes: int = 25):
    """Southward transport through the north channel faces binned by density class (MOM6 R_k) [m³ s⁻¹]."""
    edges = cases.targets(n_classes)
    tot = np.zeros(n_classes)
    idx = _window(s, t0, t1)
    for n in idx:
        v = -s["v"][n][:, -1, :] * np.nan_to_num(s["h"][n][:, -1, :]) * s["dx"][None, :]
        rho = _rho(np.nan_to_num(s["T"][n][:, -1, :]))
        k = np.clip(np.searchsorted(edges, rho) - 1, 0, n_classes - 1)
        np.add.at(tot, k.ravel(), v.ravel())
    return edges, tot / max(1, idx.size)


def thin_vs_thick(s: dict, h_thin: float = 0.01, h_thick: float = 1.0):
    """Max |u| at u-faces whose thinner side is < h_thin and at those whose both sides are ≥ h_thick (4b/4d) [m s⁻¹]."""
    h = np.nan_to_num(s["h"])
    hmin = np.minimum(h[..., 1:], h[..., :-1])
    u = np.abs(s["u"][..., 1:-1])
    thin, thick = u[hmin < h_thin], u[hmin >= h_thick]
    return float(thin.max()) if thin.size else 0.0, float(thick.max()) if thick.size else 0.0
