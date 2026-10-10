"""
Quality control of raw sea-level records and least-squares harmonic analysis.

Raw, near-real-time gauge data (IOC Sea Level Station Monitoring Facility,
which relays the SMN-UNAM network; Servicio Mareográfico Nacional) are not
quality controlled.  The checks here follow the IOC manual for sea-level QC
(IOC Manuals and Guides 14 vol. IV; UNESCO/IOC 2020, "Quality control of in
situ sea level observations", IOC MG 83) and the SELENE/GESLA practice
(Pérez et al. 2013; Haigh et al. 2023).  Each check is designed so that a
real ocean signal is not mistaken for a defect:

==========================  ===============================================
defect                      real signal kept
==========================  ===============================================
spike, 1–3 samples          tsunami / harbour seiche — coherent over many
                            samples, raises the local MAD
stuck sensor (flat line)    slack water quantised at the gauge resolution —
                            the expected tide barely changes during it
datum shift (sharp step)    storm surge — changes over hours, not minutes
clock offset / drift        lagoon / harbour lag — constant in time
==========================  ===============================================

Clock errors are found by fitting h(t) ≈ a + κ p(t − τ) per window against a
reference tide p (the gauge's own historical constants or an atlas): a
whole-hour τ is a time-zone error, a τ growing linearly is clock drift.
Co-located twin sensors (SMN radar + float) are compared directly — tide,
surge and seiches cancel in their difference, leaving only instrument
behaviour (offset, gain, lag, steps, drift).

Harmonic analysis uses the OTPS/TPXO convention of ``forcing.tides``
(h = Re + i Im, nodal corrections f, u), so its constants compare directly
with TPXO and TICON-3.  Pairs closer in frequency than the Rayleigh
criterion 2π/T need inference (Foreman 1977; Pawlowicz et al. 2002):
h_c = r h_ref with a complex ratio r taken from a reference (e.g. TPXO).

References
----------
Foreman, M.G.G. (1977). Manual for tidal heights analysis and prediction.
    Pacific Marine Science Report 77-10.
Pawlowicz, R., Beardsley, B. & Lentz, S. (2002). Classical tidal harmonic
    analysis including error estimates in MATLAB using T_TIDE. Comput.
    Geosci. 28, 929–937.
Pérez, B. et al. (2013). Sea level data quality control with SELENE.
    ESEAS-RI / Puertos del Estado.
UNESCO/IOC (2020). Quality control of in situ sea level observations: a
    review and progress towards automated quality control, vol. 1.
    IOC Manuals and Guides 83.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from forcing.tides import OMEGA, predict

_MAD_TO_SIGMA = 1.4826       # σ = 1.4826 MAD for Gaussian noise


# --------------------------------------------------------------------------
# Harmonic analysis
# --------------------------------------------------------------------------
def harmonic_analysis(t: np.ndarray, h: np.ndarray, constituents: list[str],
                      inference: dict[str, tuple[str, complex]] | None = None,
                      rayleigh: float = 1.0) -> dict:
    """
    Least-squares tidal harmonic analysis in the OTPS convention.

    Model: h(t) = h̄ + Σ_c f_c [Re h_c cos θ_c − Im h_c sin θ_c],
    θ_c = ω_c (t − 1992) + φ_c + u_c — the same basis as
    ``forcing.tides.predict``, built here by predicting unit constants.

    Parameters
    ----------
    t : datetime64 array (nt,)
    h : sea level [m], NaN where missing
    constituents : names of the constants to return
    inference : {inferred: (reference, ratio)} — h_inferred = ratio · h_reference,
        with a complex ratio from a trusted source (e.g. TPXO at the gauge).
    rayleigh : Rayleigh factor R; solved pairs must satisfy |Δω| T ≥ 2π R.

    Returns
    -------
    dict with hc (complex (nc,), m), mean [m], residual (nt,) [m, NaN where h
    is NaN], inferred (list of names).
    """
    inference = dict(inference or {})
    t = np.asarray(t, "datetime64[ns]")
    h = np.asarray(h, float)
    ok = np.isfinite(h)
    solved = [c for c in constituents if c not in inference]

    duration = float((t[ok][-1] - t[ok][0]) / np.timedelta64(1, "s"))
    for i, a in enumerate(solved):
        for b in solved[i + 1:]:
            if abs(OMEGA[a] - OMEGA[b]) * duration < 2 * np.pi * rayleigh:
                raise ValueError(f"record of {duration / 86400:.1f} d cannot separate {a} and {b}; "
                                 f"infer one of them (inference={{'{b}': ('{a}', ratio)}})")

    nc = len(constituents)
    pos = {c: k for k, c in enumerate(constituents)}
    cols = [np.ones(ok.sum())]
    for c in solved:
        for unit in (1.0, 1j):
            e = np.zeros(nc, complex)
            e[pos[c]] = unit
            for inf, (ref, r) in inference.items():
                if ref == c:
                    e[pos[inf]] = unit * r
            cols.append(predict(t[ok], e, constituents))
    A = np.column_stack(cols)
    x, *_ = np.linalg.lstsq(A, h[ok], rcond=None)

    hc = np.zeros(nc, complex)
    for k, c in enumerate(solved):
        hc[pos[c]] = x[1 + 2 * k] + 1j * x[2 + 2 * k]
    for inf, (ref, r) in inference.items():
        hc[pos[inf]] = r * hc[pos[ref]]

    resid = np.full(h.shape, np.nan)
    resid[ok] = h[ok] - A @ x
    return dict(hc=hc, mean=float(x[0]), residual=resid,
                inferred=[c for c in constituents if c in inference])


# --------------------------------------------------------------------------
# Time axis
# --------------------------------------------------------------------------
def regularize(t: np.ndarray, h: np.ndarray, step: np.timedelta64, tol: float = 0.02,
               min_gap: np.timedelta64 = np.timedelta64(30, "m")) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Put irregular raw samples on a regular time grid.

    Samples are rounded to the nearest grid time; the grid is anchored to
    multiples of ``step`` so that different sensors share grid times.  Several samples in one bin
    are averaged if they agree within ``tol`` [m] (repeated transmissions);
    otherwise the bin is set to NaN (conflicting duplicates — which one is
    right cannot be known).  Empty bins are NaN.

    Returns
    -------
    t_reg, h_reg, info with n_unsorted (samples earlier than their
    predecessor), n_duplicate (extra samples in occupied bins), n_conflict
    (bins discarded) and gaps [(start, length)] of at least ``min_gap``.
    """
    t = np.asarray(t, "datetime64[ns]")
    h = np.asarray(h, float)
    step = np.timedelta64(step, "ns")
    n_unsorted = int(np.sum(t[1:] < t[:-1]))

    epoch = np.datetime64("1970-01-01", "ns")
    t0 = epoch + np.round((t.min() - epoch) / step).astype(np.int64) * step   # grid on multiples of step
    k = np.round((t - t0) / step).astype(np.int64)
    if k.min() < 0:                                     # cannot happen with rounding, kept explicit
        t0, k = t0 + k.min() * step, k - k.min()
    s = pd.Series(h).groupby(k)
    mean, spread, count = s.mean(), s.max() - s.min(), s.size()

    n = int(k.max()) + 1
    out = np.full(n, np.nan)
    good = spread.to_numpy() <= tol
    out[mean.index.to_numpy()[good]] = mean.to_numpy()[good]

    t_reg = t0 + np.arange(n) * step
    gaps = []
    occupied = np.zeros(n, bool)
    occupied[count.index.to_numpy()] = True
    idx = np.flatnonzero(occupied)
    for a, b in zip(idx[:-1], idx[1:]):
        length = (b - a - 1) * step
        if b - a > 1 and length >= min_gap:
            gaps.append((t_reg[a + 1], length))
    info = dict(n_unsorted=n_unsorted, n_duplicate=int((count - 1).sum()),
                n_conflict=int((~good).sum()), gaps=gaps)
    return t_reg, out, info


# --------------------------------------------------------------------------
# Sample-level checks (regular series)
# --------------------------------------------------------------------------
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of the runs of True in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def flag_spikes(h: np.ndarray, expected: np.ndarray | None = None, half_window: int = 7,
                mad_window: int = 121, k: float = 6.0, min_dev: float = 0.05,
                max_run: int = 3) -> np.ndarray:
    """
    Flag isolated outliers (spikes) in a regular series.

    The deviation d = r − median₂ₙ₊₁(r) of r = h − expected from its running
    median is compared with a local robust scale σ = 1.4826 MAD(d) over
    ``mad_window`` samples: |d| > max(k σ, min_dev).  Only runs of at most
    ``max_run`` samples are spikes — a tsunami or seiche is coherent over
    many samples and raises σ locally, so it is not flagged.  ``min_dev`` [m]
    protects quantised data, whose MAD can be zero.
    """
    r = np.asarray(h, float) - (0.0 if expected is None else np.asarray(expected, float))
    s = pd.Series(r)
    d = r - s.rolling(2 * half_window + 1, center=True, min_periods=1).median().to_numpy()
    sigma = _MAD_TO_SIGMA * pd.Series(np.abs(d)).rolling(mad_window, center=True,
                                                           min_periods=1).median().to_numpy()
    cand = np.abs(d) > np.maximum(k * sigma, min_dev)
    bad = np.zeros(r.size, bool)
    for a, b in _runs(cand):
        if b - a <= max_run:
            bad[a:b] = True
    return bad


def flag_flatlines(h: np.ndarray, resolution: float, expected: np.ndarray | None = None,
                   min_change: float = 3.0, min_abs: float = 0.02,
                   max_samples: int | None = None) -> np.ndarray:
    """
    Flag a stuck sensor: runs of identical consecutive values.

    A run is flagged if the expected tide changes by more than
    max(``min_change`` × ``resolution``, ``min_abs`` [m]) during it (the sea
    moved but the sensor did not), or if it lasts more than ``max_samples``.
    ``min_abs`` keeps fine-resolution (1 mm) gauges from being judged against
    the centimetre-level error of the reference tide.
    Slack water quantised at the gauge resolution is therefore not flagged.
    NaN breaks a run.
    """
    h = np.asarray(h, float)
    same = np.zeros(h.size, bool)
    same[1:] = np.abs(np.diff(h)) < 0.5 * resolution          # NaN compares False
    bad = np.zeros(h.size, bool)
    for a, b in _runs(same):
        a -= 1                                                 # include the first sample of the run
        hit = max_samples is not None and b - a > max_samples
        if expected is not None:
            e = np.asarray(expected, float)[a:b]
            hit |= np.nanmax(e) - np.nanmin(e) > max(min_change * resolution, min_abs)
        if hit:
            bad[a:b] = True
    return bad


def detect_steps(r: np.ndarray, short: int = 10, long: int = 1490,
                 threshold: float = 0.05, k_noise: float = 5.0) -> list[dict]:
    """
    Detect datum shifts in a residual series (observation − tide, or the
    difference between twin sensors).

    At each valid sample i the median of the ``long`` samples after i minus
    that of the ``long`` before (D_L) measures a persistent level change,
    and the same with ``short`` samples (D_S) its sharpness.  A step needs
    |D_L| > max(threshold, k_noise σ_S) and the same change at the short and
    intermediate (6 × ``short``) scales, |D_S − D_L| and |D_M − D_L| < ½|D_L|, with σ_S
    the robust spread (1.4826 MAD) of D_S over the record — the short-scale
    noise (seiches, waves, quantisation) that could mimic a step — and a
    level that is stable on both sides: |D_M| > 3σ_local, σ_local the larger
    robust spread of the 6 × ``short`` samples before and after.  Energetic
    shelf oscillations with sharp fronts (e.g. Progreso) are thus not steps;
    a shift hidden in such noise is left to the twin-sensor comparison.  The default
    ``long`` is one tidal day at 1-min sampling (24.84 h), so that residual
    tides average out of D_L.  A storm
    surge changes the level over hours (small D_S), a spike is not
    persistent (small D_L).  Gaps are skipped, so a shift during a
    maintenance gap is found at the first sample after it.

    Returns
    -------
    list of dict(index, size [m], gap_samples) — ``index`` into ``r``,
    ``size`` the difference of medians over 6 × ``short`` samples on each
    side (less biased by slow signals than D_L), ``gap_samples`` missing
    samples just before the step.
    """
    r = np.asarray(r, float)
    idx = np.flatnonzero(np.isfinite(r))
    v = pd.Series(r[idx])

    def before(n: int) -> np.ndarray:                         # median of v[i-n:i]
        return v.rolling(n, min_periods=n).median().shift(1).to_numpy()

    def after(n: int) -> np.ndarray:                          # median of v[i:i+n]
        return v[::-1].rolling(n, min_periods=n).median()[::-1].to_numpy()

    dl = after(long) - before(long)
    ds = after(short) - before(short)
    dsize = after(6 * short) - before(6 * short)

    def spread(x: pd.Series) -> np.ndarray:                  # robust σ over 6·short samples
        m = x.rolling(6 * short, min_periods=6 * short).median()
        return _MAD_TO_SIGMA * (x - m).abs().rolling(6 * short, min_periods=6 * short).median().to_numpy()

    s_before = pd.Series(spread(v)).shift(1).to_numpy()
    s_after = spread(v[::-1].reset_index(drop=True))[::-1]
    local = np.fmax(s_before, s_after)
    sigma = _MAD_TO_SIGMA * np.nanmedian(np.abs(ds - np.nanmedian(ds))) if np.isfinite(ds).any() else 0.0
    with np.errstate(invalid="ignore"):
        cand = ((np.abs(dl) > max(threshold, k_noise * sigma)) & (np.abs(ds - dl) < 0.5 * np.abs(dl))
                & (np.abs(dsize - dl) < 0.5 * np.abs(dl)) & (np.abs(dsize) > 3.0 * local))

    # Medians are flat for a few samples around a step; the difference of
    # short means peaks at it and locates the step.
    dm = (v[::-1].rolling(short, min_periods=short).mean()[::-1].to_numpy()
          - v.rolling(short, min_periods=short).mean().shift(1).to_numpy())
    steps = []
    for a, b in _runs(cand):
        i = a + int(np.nanargmax(np.abs(dm[a:b])))
        gap = int(idx[i] - idx[i - 1] - 1) if i > 0 else 0
        steps.append(dict(index=int(idx[i]), size=float(dsize[i]), gap_samples=gap))
    return steps


# --------------------------------------------------------------------------
# Clock and twin-sensor checks
# --------------------------------------------------------------------------
def _lag_fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each row x of X (lagged references) fit y ≈ a + κ x by least squares
    over the samples finite in y.  Returns κ and the residual variance per row.
    """
    ok = np.isfinite(y)
    y = y[ok] - y[ok].mean()
    X = X[:, ok]
    X = X - X.mean(axis=1, keepdims=True)
    sxx = np.einsum("ij,ij->i", X, X)
    sxy = X @ y
    kappa = sxy / sxx
    resid = (y @ y - kappa * sxy) / y.size
    return kappa, resid


def _best_lag(kappa: np.ndarray, resid: np.ndarray, lags: np.ndarray) -> tuple[float, float]:
    """Lag of minimum residual, refined by a parabola through its neighbours."""
    j = int(np.argmin(resid))
    lag = float(lags[j])
    if 0 < j < len(lags) - 1:
        r0, r1, r2 = resid[j - 1], resid[j], resid[j + 1]
        den = r0 - 2 * r1 + r2
        if np.isfinite(den) and den > 0:
            lag += 0.5 * (r0 - r2) / den * (lags[1] - lags[0])
    return lag, float(kappa[j])


def clock_offsets(t: np.ndarray, h: np.ndarray, reference: Callable[[np.ndarray], np.ndarray],
                  window: np.timedelta64 = np.timedelta64(3, "D"),
                  max_lag: np.timedelta64 = np.timedelta64(13, "h"),
                  min_valid: float = 0.5, positive_gain: bool = False) -> dict:
    """
    Clock offset τ and gain κ per window: h(t) ≈ a + κ p(t − τ).

    ``reference`` returns the reference tide p at given times (e.g.
    ``lambda tt: predict(tt, hc_ticon, cons)``) and is evaluated on the
    regular grid of ``t`` extended by ``max_lag``.  Positive τ: the gauge
    clock is ahead (its record is late relative to the reference).  Windows
    with fewer than ``min_valid`` valid samples give NaN.  ``positive_gain``
    restricts the fit to κ > 0, which removes the half-period ambiguity
    p(t − T_M2/2) ≈ −p(t) of a semidiurnal tide — use it when the reference
    already has the gauge's sign (e.g. its own harmonic constants).

    Returns
    -------
    dict of arrays: t_center, tau_minutes, gain, n_valid and skill — the
    fraction of the window variance explained by the fit (low when the tide
    is weak against surge or noise, and τ is then unreliable).
    """
    t = np.asarray(t, "datetime64[ns]")
    h = np.asarray(h, float)
    step = t[1] - t[0]
    L = int(np.timedelta64(max_lag, "ns") // step)
    W = int(np.timedelta64(window, "ns") // step)
    p = np.asarray(reference(t[0] + np.arange(-L, t.size + L) * step), float)
    lags = np.arange(-L, L + 1)

    centers, taus, gains, nval, skill = [], [], [], [], []
    for i0 in range(0, t.size - W + 1, W):
        y = h[i0:i0 + W]
        centers.append(t[i0] + W * step // 2)
        nval.append(int(np.isfinite(y).sum()))
        if nval[-1] < min_valid * W:
            taus.append(np.nan)
            gains.append(np.nan)
            skill.append(np.nan)
            continue
        # h(t_i) vs p(t_i − l·step): row l is p[i0 + L − l : … + W]
        X = np.lib.stride_tricks.sliding_window_view(p[i0:i0 + 2 * L + W], W)[::-1]
        kappa, resid = _lag_fit(y, X)
        if positive_gain:
            resid = np.where(kappa > 0, resid, np.inf)
        lag, gain = _best_lag(kappa, resid, lags)
        taus.append(lag * step / np.timedelta64(1, "m"))
        gains.append(gain)
        skill.append(1.0 - float(resid.min()) / float(np.nanvar(y)))
    return dict(t_center=np.array(centers, "datetime64[ns]"), tau_minutes=np.array(taus),
                gain=np.array(gains), n_valid=np.array(nval), skill=np.array(skill))


def compare_twins(h1: np.ndarray, h2: np.ndarray, step: np.timedelta64,
                  max_lag: np.timedelta64 = np.timedelta64(30, "m"),
                  diff_lag: np.timedelta64 = np.timedelta64(1, "h"),
                  smooth: np.timedelta64 = np.timedelta64(1, "h"), **step_kw) -> dict:
    """
    Compare two co-located sensors on the same regular grid.

    Fits h2(t) ≈ a + κ h1(t − τ) (τ > 0: sensor 2 lags sensor 1); κ and τ
    come from the increments over ``diff_lag`` (Δh2 ≈ κ Δh1), which are
    insensitive to offsets and datum steps of either sensor (increments
    deviating > 5σ from a first fit are dropped and the fit repeated).  Both
    series are first smoothed with a centred running mean over ``smooth``:
    seiches and waves seen by only one sensor (a radar sees them, a float in
    a stilling well does not) would otherwise act as noise in the regressor
    and dilute κ (errors-in-variables bias); the same filter on both sensors
    leaves κ and τ of the tide unchanged.  Then examines
    the aligned difference e = a + κ h1(t − τ) − h2, from which tide, surge
    and seiches have cancelled: datum steps (``detect_steps``) and linear
    drift (slope of the daily medians of e).

    Returns
    -------
    dict with lag_minutes, gain κ, offset (median of h1 − h2) [m],
    rms_diff (std of e) [m], steps (list, sizes in e), drift_m_per_day.
    """
    h1 = np.asarray(h1, float)
    h2 = np.asarray(h2, float)
    step = np.timedelta64(step, "ns")
    L = int(np.timedelta64(max_lag, "ns") // step)
    lags = np.arange(-L, L + 1)
    n = h1.size - 2 * L
    y = h2[L:L + n]
    X = np.lib.stride_tricks.sliding_window_view(h1, n)[::-1]       # row l: h1(t − l)

    S = max(1, int(np.timedelta64(smooth, "ns") // step))
    lp1 = pd.Series(h1).rolling(S, center=True, min_periods=max(1, S // 2)).mean().to_numpy()
    lp2 = pd.Series(h2).rolling(S, center=True, min_periods=max(1, S // 2)).mean().to_numpy()
    ys = lp2[L:L + n]
    Xs = np.lib.stride_tricks.sliding_window_view(lp1, n)[::-1]

    D = int(np.timedelta64(diff_lag, "ns") // step)
    dy = ys[D:] - ys[:-D]
    dX = Xs[:, D:] - Xs[:, :-D]
    ok = np.isfinite(dy) & np.all(np.isfinite(dX), axis=0)
    for _ in range(2):                         # 2nd pass drops increments spanning steps/spikes
        kappa, resid = _lag_fit(np.where(ok, dy, np.nan), np.where(ok, dX, 0.0))
        lag, gain = _best_lag(kappa, resid, lags)
        e = dy - gain * dX[int(np.round(lag)) + L]
        mad = np.nanmedian(np.abs(e[ok] - np.nanmedian(e[ok])))
        ok &= np.abs(e - np.nanmedian(e[ok])) <= 5 * _MAD_TO_SIGMA * mad
    ok = np.isfinite(y) & np.isfinite(X[int(np.round(lag)) + L])

    j = int(np.round(lag)) + L
    xs = X[j]
    a = np.nanmean(y[ok]) - gain * np.nanmean(xs[ok])
    e = np.full(h2.size, np.nan)
    e[L:L + n] = a + gain * xs - y

    per_day = int(np.timedelta64(1, "D") // step)
    days = pd.Series(e).groupby(np.arange(e.size) // per_day).median().dropna()
    drift = float(np.polyfit(days.index.to_numpy(), days.to_numpy(), 1)[0]) if len(days) > 2 else np.nan
    return dict(lag_minutes=lag * step / np.timedelta64(1, "m"), gain=gain,
                offset=float(np.nanmedian(h1 - h2)), rms_diff=float(np.nanstd(e)),
                steps=detect_steps(e, **step_kw), drift_m_per_day=drift)


def clock_changes(t_center: np.ndarray, tau_minutes: np.ndarray, min_windows: int = 3,
                  min_jump: float = 15.0, min_drift: float = 10.0, z_crit: float = 5.0) -> dict:
    """
    Decide whether a series of window clock offsets τ (``clock_offsets``)
    contains a persistent jump or a drift, beyond the window-to-window
    scatter caused by surge and noise.

    Jump: the split with the largest standardised difference of medians,
    z = |Δ| / (1.2533 σ √(1/n₁ + 1/n₂)), σ = 1.4826 MAD of the deviations
    from the two medians; a jump needs z > ``z_crit``, |Δ| > ``min_jump``
    [min] and ``min_windows`` windows on each side.  Drift: least-squares
    slope with t-statistic > ``z_crit`` and total change > ``min_drift``
    [min].  When both qualify, the model with the smaller residual wins.

    Returns
    -------
    dict(jump, jump_minutes, jump_time, jump_z, drift, drift_min_per_day, drift_z)
    """
    t_center = np.asarray(t_center, "datetime64[ns]")
    tau = np.asarray(tau_minutes, float)
    ok = np.isfinite(tau)
    t_c, y = t_center[ok], tau[ok]
    out = dict(jump=False, jump_minutes=np.nan, jump_time=None, jump_z=0.0,
               drift=False, drift_min_per_day=np.nan, drift_z=0.0)
    n = y.size
    if n < 2 * min_windows:
        return out

    best, sse_jump = None, np.inf
    for k in range(min_windows, n - min_windows + 1):
        m1, m2 = np.median(y[:k]), np.median(y[k:])
        dev = np.concatenate([y[:k] - m1, y[k:] - m2])
        sigma = max(_MAD_TO_SIGMA * np.median(np.abs(dev)), 1e-9)
        z = abs(m2 - m1) / (1.2533 * sigma * np.sqrt(1 / k + 1 / (n - k)))
        if best is None or z > best[0]:
            best, sse_jump = (z, m2 - m1, k), float(np.sum(dev ** 2))
    z, delta, k = best
    out.update(jump_minutes=float(delta), jump_time=t_c[k], jump_z=float(z))
    jump = z > z_crit and abs(delta) > min_jump

    days = (t_c - t_c[0]) / np.timedelta64(1, "D")
    coef, res, *_ = np.polyfit(days, y, 1, full=True)
    sse_lin = float(res[0]) if res.size else 0.0
    se = np.sqrt(sse_lin / max(n - 2, 1) / np.sum((days - days.mean()) ** 2))
    tz = abs(coef[0]) / se if se > 0 else np.inf
    out.update(drift_min_per_day=float(coef[0]), drift_z=float(tz))
    drift = tz > z_crit and abs(coef[0]) * days[-1] > min_drift

    if jump and drift:
        jump, drift = (True, False) if sse_jump < sse_lin else (False, True)
    out.update(jump=bool(jump), drift=bool(drift))
    return out
