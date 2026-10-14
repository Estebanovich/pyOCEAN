"""
Metrics of the four-model benchmark (spec §A4), identical for every model, on the common series (series.py).

RPE (Winters et al. 1995; Ilicak et al. 2012, eq. 1–2): RPE = g ∫ ρ* z dV with ρ* the adiabatically sorted density
and z the height above the deepest point of each model's own bed; computed by
diagnostics.mixing.reference_potential_energy on the true cell volumes h·Δx·Δy and the exact hypsometry of the bed
(step changes of area at each column's bed height), so z-levels with partial cells, σ and ALE layers are treated alike.
dRPE/dt is per unit surface area [W m⁻²] as in Ilicak (Fig. 6) and Petersen (2015).
"""
from __future__ import annotations

import numpy as np

from diagnostics.mixing import reference_potential_energy
from validation.ilicak_benchmark import cases

BENJAMIN = 0.5 * np.sqrt(cases.G * 5.0 / cases.RHO0 * 20.0)     # Benjamin (1968) u_f = ½ √(g′H) [m s⁻¹]


def hypsometry(H: np.ndarray, dx: np.ndarray, dy: np.ndarray, top: float = 10.0,
               n_fine: int = 50000) -> tuple[np.ndarray, np.ndarray]:
    """(z, A(z)): heights above the deepest bed [m, increasing] and the horizontal area [m²] at those heights, with
    each column's area added at its bed height (a 1e-9 m ramp makes the step exact for the trapezoid integral);
    the top is ``top`` m above the rest surface.  Every segment is refined to steps ≤ (H_max + top)/n_fine because
    reference_potential_energy interpolates the moment ∫ z A dz linearly in volume (relative RPE error
    ≈ (Δz/H)²/8 ≈ 5e-11 at n_fine = 50000; with the two points of a flat basin it would be 50 % off)."""
    Hc = np.nan_to_num(np.asarray(H, float))
    area = np.asarray(dy, float)[:, None] * np.asarray(dx, float)[None, :]
    wet = Hc > 0
    hmax = Hc[wet].max()
    bed = hmax - Hc[wet]
    a = area[wet]
    levels = np.unique(bed)
    z, A = [0.0], [a[bed <= levels[0]].sum()]
    for b in levels[1:]:
        z += [b - 1e-9, b]
        A += [A[-1], a[bed <= b].sum()]
    z.append(hmax + top)
    A.append(A[-1])
    step = (hmax + top) / n_fine
    zf, af = [z[0]], [A[0]]
    for z0, z1, a0, a1 in zip(z[:-1], z[1:], A[:-1], A[1:]):
        n = max(1, int(np.ceil((z1 - z0) / step)))
        zf += list(np.linspace(z0, z1, n + 1)[1:])
        af += list(np.linspace(a0, a1, n + 1)[1:])
    return np.asarray(zf), np.asarray(af)


def rpe(s: dict) -> np.ndarray:
    """RPE [J] of every record."""
    zA = hypsometry(s["H"], s["dx"], s["dy"])
    vol = s["h"] * (s["dy"][:, None] * s["dx"][None, :])[None, None]
    rho = cases.density(s["T"])
    return np.array([reference_potential_energy(np.where(vol[n] > 0, rho[n], np.nan), vol[n], hypsometry=zA)
                     for n in range(s["t"].size)])


def drpe_dt(s: dict, rpe_t: np.ndarray, t0: float, t1: float) -> float:
    """Mean dRPE/dt between the records nearest t0 and t1, per unit surface area [W m⁻²]."""
    i0, i1 = (int(np.argmin(np.abs(s["t"] - t))) for t in (t0, t1))
    area = float(np.sum(np.isfinite(np.asarray(s["H"], float)) * (s["dy"][:, None] * s["dx"][None, :])))
    return float((rpe_t[i1] - rpe_t[i0]) / (s["t"][i1] - s["t"][i0]) / area)


def t_range(s: dict, h_min: float = 1e-3) -> tuple[float, float]:
    """(min, max) of T over all records in cells thicker than h_min [m] (new extrema = non-monotone advection)."""
    T = np.where(s["h"] > h_min, s["T"], np.nan)
    return float(np.nanmin(T)), float(np.nanmax(T))


def _front(x: np.ndarray, thick: np.ndarray, start: float, sign: int, thr: float) -> float:
    """Distance from ``start`` of the farthest column (east if sign > 0) whose ``thick`` exceeds ``thr``."""
    sel = (sign * (x - start) > 0) & (thick > thr)
    return float(np.max(sign * (x[sel] - start))) if sel.any() else 0.0


def lock_fronts(s: dict, t_fit: tuple = (2 * 3600.0, 8 * 3600.0)) -> dict:
    """Front speeds [m s⁻¹] of the lock exchange (spec §A4; as validation/mom6_lock_exchange): dense water (T < 17.5)
    thicker than 1 m (nose) / 5 m (body) east of the gate, light water (T > 17.5) likewise west of it; speed =
    least-squares slope of the position over t_fit."""
    x, gate = s["x"], 32e3
    sel = (s["t"] >= t_fit[0]) & (s["t"] <= t_fit[1])
    out = {}
    for name, cold, sign, thr in (("dense_nose", True, 1, 1.0), ("light_nose", False, -1, 1.0),
                                  ("dense_body", True, 1, 5.0), ("light_body", False, -1, 5.0)):
        pos = []
        for n in np.flatnonzero(sel):
            T = s["T"][n, :, 0, :]
            mask = (T < 17.5) if cold else (T > 17.5)
            thick = np.where(mask, s["h"][n, :, 0, :], 0.0).sum(0)
            pos.append(_front(x, thick, gate, sign, thr))
        out[name] = float(np.polyfit(s["t"][sel], pos, 1)[0])
    return out


def _centres(s: dict, n: int) -> np.ndarray:
    """Cell-centre depths [m, positive down] of record n (surface-first layers below η)."""
    h = s["h"][n]
    top = np.concatenate([np.zeros((1,) + h.shape[1:]), np.cumsum(h, 0)[:-1]]) - s["eta"][n][None]
    return top + 0.5 * h


def overflow_descent(s: dict, T_mid: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass (x_c [m], depth_c [m]) of the dense water (T < T_mid) in every record."""
    xs, ds = [], []
    for n in range(s["t"].size):
        V = s["h"][n] * (s["dy"][:, None] * s["dx"][None, :])[None]
        w = np.where(np.nan_to_num(s["T"][n], nan=np.inf) < T_mid, V, 0.0)
        xs.append(float((w * s["x"][None, None, :]).sum() / w.sum()))
        ds.append(float((w * _centres(s, n)).sum() / w.sum()))
    return np.array(xs), np.array(ds)


def overflow_cold_centroid(s: dict, T_amb: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Centroid (x [m], depth [m]) of the cold anomaly (T_amb − T)⁺ in every record: unlike the T < 15 °C centre of
    mass it stays defined when the dense water has been mixed above the threshold."""
    xs, ds = [], []
    for n in range(s["t"].size):
        V = s["h"][n] * (s["dy"][:, None] * s["dx"][None, :])[None]
        w = V * np.clip(T_amb - np.nan_to_num(s["T"][n], nan=T_amb), 0.0, None)
        xs.append(float((w * s["x"][None, None, :]).sum() / w.sum()))
        ds.append(float((w * _centres(s, n)).sum() / w.sum()))
    return np.array(xs), np.array(ds)


def iw_amplitude(s: dict, x0: float = 125e3, T_iso: float | None = None) -> np.ndarray:
    """Displacement [m, positive up] of the isotherm T_iso (default: the mid-depth background value 15.1 °C) in the
    column nearest x0, relative to record 0; linear interpolation between cell centres."""
    T_iso = 15.1 if T_iso is None else T_iso
    i = int(np.argmin(np.abs(s["x"] - x0)))
    depth = []
    for n in range(s["t"].size):
        zc = _centres(s, n)[:, 0, i]
        T = s["T"][n, :, 0, i]
        ok = s["h"][n, :, 0, i] > 1e-3
        zc, T = zc[ok], T[ok]
        depth.append(float(np.interp(T_iso, T[::-1], zc[::-1])))     # T decreases with depth
    depth = np.array(depth)
    return depth[0] - depth


def max_speed(s: dict, h_min: float = 0.01) -> np.ndarray:
    """max |u|, |v| [m s⁻¹] of every record over faces whose two neighbouring cells are at least h_min [m] thick:
    velocities in vanished layers (MOM6 keeps 1-mm layers along a z* slope) carry no mass."""
    h = s["h"]
    hu = np.zeros(s["u"].shape)
    hu[..., 1:-1] = np.minimum(h[..., 1:], h[..., :-1])
    hv = np.zeros(s["v"].shape)
    hv[..., 1:-1, :] = np.minimum(h[..., 1:, :], h[..., :-1, :])
    u = np.where(hu >= h_min, np.abs(s["u"]), 0.0).reshape(s["t"].size, -1).max(1)
    v = np.where(hv >= h_min, np.abs(s["v"]), 0.0).reshape(s["t"].size, -1).max(1)
    return np.maximum(u, v)
