"""
Metrics of the DOME benchmark (spec 2026-10-14-dome-overflow §7; Legg et al. 2006 eq. 11–13, 18, Fig. 15), the same
functions for every model on the common series (validation/dome/series.py; layers surface first, h ≥ 0, land NaN).
x is along the slope (the plume turns west: transports westward positive), y across it.
"""
from __future__ import annotations

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
    with np.errstate(all="ignore"):
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
    zA = hypsometry(s["H"], s["dx"], s["dy"])
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
