"""
Quality control of in-situ temperature/salinity profiles and fixed-depth records (gliders, wire-following
profilers, moored CTDs, bottom-temperature loggers) before they are used as truth in a benchmark, and the profile
metrics shared by observations and models (mixed-layer depth, thermocline depth).

The checks follow the Argo real-time tests and the U.S. IOOS QARTOD manual, applied to the data as they come
(provider flags are honoured where present, our tests decide where they are absent or "not evaluated"):

==========================  ======================================================================================
check                        reference / rule
==========================  ======================================================================================
provider flags               QARTOD 1 pass, 2 not evaluated (kept for our tests), 3 suspect, 4 fail, 9 missing
                             (U.S. IOOS 2020, *Manual for real-time QC of in-situ temperature and salinity data*,
                             v2.1); IOOS glider DAC / Argo scale 0 no QC, 1 good, 2 probably good, 3 bad but
                             correctable, 4 bad, 5 changed, 8 interpolated, 9 missing (Argo QC manual, Wong et al.
                             2023, doi:10.13155/33951) [verificar the version]
gross range                  regional limits passed by the caller
spike                        Argo test 9: |x₂ − (x₃ + x₁)/2| − |(x₃ − x₁)/2| above 6 °C / 0.9 PSU shallower than
                             500 dbar and 2 °C / 0.3 PSU deeper (Argo QC manual, test 9)
stuck value                  Argo test 13: one value over the whole profile
density inversion            Argo test 14: σ₀ lighter than the bin above by more than 0.03 kg m⁻³
out of water                 conductivity below a floor (in-air or bubble samples of a glider's CT cell)
position / time              Argo tests 3–5: speed between consecutive profiles, duplicate profiles
off-station pressure         a moored sensor away from its record-median pressure (deployment/recovery, knock-down)
cross-check                  bias and drift of an unflagged platform against nearby accepted profiles
==========================  ======================================================================================

Mixed-layer depth: the σ₀ threshold method of de Boyer Montégut et al. (2004, doi:10.1029/2004JC002378), here
with a 5 m reference depth (shelf summer mixed layers are 5–20 m deep, so their 10 m reference is too deep) and
the same 0.03 kg m⁻³ threshold.  Thermocline depth: depth of the maximum −∂T/∂z after a running mean.

Array conventions: observation-level arrays are 1-D with a profile index ``pid`` (rows of one profile adjacent, in
sampling order); binned profiles are (n_prof, n_bins) on depth bins (positive down, m).
"""
from __future__ import annotations

import numpy as np

#: accepted provider flag values per scheme (NaN = no provider flag → our tests decide)
ACCEPT = {"qartod": (1, 2), "dac": (0, 1, 2, 5)}
EARTH_KM = 6371.0


def provider_ok(flag: np.ndarray, scheme: str) -> np.ndarray:
    """True where the provider flag accepts the sample (or there is no flag).

    ``scheme`` 'qartod' accepts 1 (pass) and 2 (not evaluated: left to our tests); 'dac' (IOOS glider DAC / Argo)
    accepts 0 (no QC done), 1, 2 and 5 (value changed by the provider).  8 (interpolated) is rejected: it is not a
    measurement."""
    if scheme not in ACCEPT:
        raise ValueError(f"unknown flag scheme {scheme!r}; use one of {sorted(ACCEPT)}")
    f = np.asarray(flag, float)
    return ~np.isfinite(f) | np.isin(f, ACCEPT[scheme])


def gross_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """True where a finite value is outside [lo, hi]."""
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(x) & ((x < lo) | (x > hi))


def out_of_water(c: np.ndarray, c_min: float = 2.0) -> np.ndarray:
    """True where conductivity [S m⁻¹] is below ``c_min``: in-air or bubble samples of a CT cell (sea water on the
    shelf has C > 3 S m⁻¹).  The temperature of such a sample is air or wet-sensor temperature, so both T and S go."""
    c = np.asarray(c, float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(c) & (c < c_min)


def vertical_spike(x: np.ndarray, p: np.ndarray, pid: np.ndarray, thr_shallow: float, thr_deep: float,
                   p_split: float = 500.0) -> np.ndarray:
    """Argo spike test (test 9) on adjacent finite samples of the same profile.

    V = |x₂ − (x₃ + x₁)/2| − |(x₃ − x₁)/2| for each sample x₂ with finite neighbours x₁, x₃ in the same profile;
    flagged where V > ``thr_shallow`` (p < ``p_split`` dbar) or V > ``thr_deep`` (deeper)."""
    x = np.asarray(x, float)
    out = np.zeros(x.size, bool)
    idx = np.flatnonzero(np.isfinite(x) & np.isfinite(p))
    if idx.size < 3:
        return out
    xs, ps, gs = x[idx], np.asarray(p, float)[idx], np.asarray(pid)[idx]
    same = (gs[1:-1] == gs[:-2]) & (gs[1:-1] == gs[2:])
    v = np.abs(xs[1:-1] - 0.5 * (xs[2:] + xs[:-2])) - np.abs(0.5 * (xs[2:] - xs[:-2]))
    thr = np.where(ps[1:-1] < p_split, thr_shallow, thr_deep)
    out[idx[1:-1]] = same & (v > thr)
    return out


def split_legs(t: np.ndarray, p: np.ndarray, gap: np.timedelta64 = np.timedelta64(30, "m"),
               min_span: float = 10.0, deadband: float = 0.5) -> np.ndarray:
    """Leg (single up or down cast) index of each sample of a profiling record that is not split into profiles
    (e.g. the 1-min binned rows of an OOI wire-following profiler, several depths per minute).

    The mean pressure per time stamp is walked in time order; a new leg starts at the turning point where the
    direction (changes larger than ``deadband`` dbar) reverses, and after a gap longer than ``gap``.  Parked
    stretches stay in the current leg.  Legs spanning less than ``min_span`` dbar get −1.

    Returns int (n,) leg numbers 0, 1, … in time order, −1 for dropped samples."""
    t = np.asarray(t, "datetime64[ns]")
    p = np.asarray(p, float)
    ut, inv = np.unique(t, return_inverse=True)
    pm = np.bincount(inv, weights=np.nan_to_num(p), minlength=ut.size) / np.maximum(
        np.bincount(inv, weights=np.isfinite(p), minlength=ut.size), 1)
    leg = np.zeros(ut.size, np.int64)
    cur, cur_dir = 0, 0
    big = np.diff(ut) > np.timedelta64(gap, "ns")
    d = np.diff(pm)
    for k in range(d.size):
        if big[k]:
            cur, cur_dir = cur + 1, 0
            leg[k + 1] = cur
            continue
        s = 0 if abs(d[k]) < deadband else (1 if d[k] > 0 else -1)
        if s and cur_dir and s != cur_dir:
            cur += 1
            leg[k] = cur                                   # the turning point starts the new leg
        if s:
            cur_dir = s
        leg[k + 1] = cur
    # span per leg; renumber the kept ones
    lo = np.full(cur + 1, np.inf)
    hi = np.full(cur + 1, -np.inf)
    np.minimum.at(lo, leg, pm)
    np.maximum.at(hi, leg, pm)
    keep = (hi - lo) >= min_span
    new = np.full(cur + 1, -1, np.int64)
    new[keep] = np.arange(keep.sum())
    return new[leg][inv]


def bin_profiles(pid: np.ndarray, z: np.ndarray, x: np.ndarray, edges: np.ndarray, n_prof: int) -> np.ndarray:
    """Mean of the finite samples of each profile in each depth bin [edges[b], edges[b+1]) → (n_prof, n_bins),
    NaN in empty bins.  Samples with pid < 0 or outside the bins are ignored."""
    nb = edges.size - 1
    b = np.searchsorted(edges, z, side="right") - 1
    ok = np.isfinite(x) & np.isfinite(z) & (b >= 0) & (b < nb) & (np.asarray(pid) >= 0)
    flat = np.asarray(pid)[ok] * nb + b[ok]
    s = np.bincount(flat, weights=np.asarray(x, float)[ok], minlength=n_prof * nb)
    c = np.bincount(flat, minlength=n_prof * nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out.reshape(n_prof, nb)


def density_inversion(sig: np.ndarray, thr: float = 0.03) -> np.ndarray:
    """Argo test 14 on binned σ₀ (n_prof, n_bins), bins ordered from the surface down: True where a bin is lighter
    than the nearest finite bin above it by more than ``thr`` kg m⁻³."""
    sig = np.atleast_2d(np.asarray(sig, float))
    bad = np.zeros(sig.shape, bool)
    last = np.full(sig.shape[0], np.nan)
    for k in range(sig.shape[1]):
        cur = sig[:, k]
        with np.errstate(invalid="ignore"):
            bad[:, k] = np.isfinite(cur) & np.isfinite(last) & (cur < last - thr)
        last = np.where(np.isfinite(cur), cur, last)
    return bad


def stuck_profiles(x: np.ndarray, min_n: int = 10) -> np.ndarray:
    """Argo test 13: True for binned profiles (n_prof, n_bins) with at least ``min_n`` finite bins that all hold
    the same value."""
    x = np.atleast_2d(np.asarray(x, float))
    n = np.isfinite(x).sum(axis=1)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        rng = np.nanmax(x, axis=1) - np.nanmin(x, axis=1)
    return (n >= min_n) & (rng == 0.0)


def haversine_km(lat1: np.ndarray | float, lon1: np.ndarray | float, lat2: np.ndarray | float,
                 lon2: np.ndarray | float) -> np.ndarray | float:
    """Great-circle distance [km] on a sphere of radius 6371 km."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def duplicate_profiles(t: np.ndarray, lat: np.ndarray, lon: np.ndarray, pos_tol: float = 1e-5) -> np.ndarray:
    """True for a profile whose time and position repeat an earlier one (Argo duplicate test)."""
    seen: set = set()
    out = np.zeros(len(t), bool)
    for k, (tt, la, lo) in enumerate(zip(np.asarray(t, "datetime64[s]"), lat, lon)):
        key = (int(tt.astype(np.int64)), round(float(la) / pos_tol), round(float(lo) / pos_tol))
        out[k] = key in seen
        seen.add(key)
    return out


def speed_flags(t: np.ndarray, lat: np.ndarray, lon: np.ndarray, vmax: float = 3.0) -> np.ndarray:
    """Argo speed test: True for a profile reached from the last accepted profile at more than ``vmax`` m s⁻¹
    (a glider moves at ≤ 0.4 m s⁻¹ through the water; 3 m s⁻¹ allows for the Gulf Stream)."""
    t = np.asarray(t, "datetime64[ns]")
    out = np.zeros(t.size, bool)
    last = None
    for k in range(t.size):
        if not (np.isfinite(lat[k]) and np.isfinite(lon[k])):
            out[k] = True
            continue
        if last is not None:
            dt = abs((t[k] - t[last]) / np.timedelta64(1, "s"))
            dist = haversine_km(lat[last], lon[last], lat[k], lon[k]) * 1e3
            if dist > vmax * max(dt, 60.0):
                out[k] = True
                continue
        last = k
    return out


def pressure_off_station(p: np.ndarray, tol: float, window: int | None = None) -> np.ndarray:
    """True where a fixed-depth sensor's pressure is more than ``tol`` dbar from its median: out of the water
    (deployment, recovery), dragged, or wrong units.  The median is the record median, or a centred running median
    of ``window`` samples, which keeps successive deployments at different depths (a mooring turnaround)."""
    p = np.asarray(p, float)
    if window is None:
        ref = np.nanmedian(p)
    else:
        import pandas as pd
        ref = pd.Series(p).rolling(window, center=True, min_periods=1).median().to_numpy()
    with np.errstate(invalid="ignore"):
        return np.isfinite(p) & (np.abs(p - ref) > tol)


def _interp_finite(z: np.ndarray, x: np.ndarray, zq: float) -> float:
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.nan
    return float(np.interp(zq, z[ok], x[ok], left=np.nan, right=np.nan))


def mixed_layer_depth(z: np.ndarray, sig: np.ndarray, z_ref: float = 5.0, dsig: float = 0.03,
                      z_top_max: float = 7.0) -> np.ndarray:
    """Mixed-layer depth [m] of binned σ₀ profiles (n_prof, n_bins) on bin centres ``z`` (positive down).

    MLD = the depth where σ₀ first exceeds σ₀(z_r) + ``dsig`` below z_r, linearly interpolated between bins
    (de Boyer Montégut et al. 2004, threshold method), z_r = max(``z_ref``, shallowest finite bin).  NaN when the
    shallowest finite bin is deeper than ``z_top_max`` (the profile misses the surface layer) or when the threshold
    is never reached (the profile does not reach the base of the mixed layer)."""
    z = np.asarray(z, float)
    sig = np.atleast_2d(np.asarray(sig, float))
    out = np.full(sig.shape[0], np.nan)
    for n, s in enumerate(sig):
        ok = np.isfinite(s)
        if ok.sum() < 3 or z[ok][0] > z_top_max:
            continue
        zr = max(z_ref, z[ok][0])
        ref = _interp_finite(z, s, zr)
        if not np.isfinite(ref):
            continue
        zz, ss = z[ok], s[ok]
        below = zz > zr
        hit = np.flatnonzero(below & (ss > ref + dsig))
        if hit.size == 0:
            continue
        k = hit[0]
        z0, s0 = (zr, ref) if k == 0 or zz[k - 1] < zr else (zz[k - 1], ss[k - 1])
        out[n] = z0 + (ref + dsig - s0) * (zz[k] - z0) / (ss[k] - s0)
    return out


def thermocline_depth(z: np.ndarray, t: np.ndarray, smooth: int = 5, z_max: float = 200.0) -> np.ndarray:
    """Depth [m] of the maximum −∂T/∂z of binned temperature profiles (n_prof, n_bins) after a ``smooth``-bin
    running mean, searched above ``z_max``; NaN with fewer than ``smooth`` + 2 finite bins.  Gradients are taken
    between adjacent finite bins only (no gradient across a gap)."""
    z = np.asarray(z, float)
    t = np.atleast_2d(np.asarray(t, float))
    kern = np.ones(smooth) / smooth
    out = np.full(t.shape[0], np.nan)
    sel = z <= z_max
    zs = z[sel]
    for n, x in enumerate(t[:, sel]):
        ok = np.isfinite(x)
        if ok.sum() < smooth + 2:
            continue
        num = np.convolve(np.where(ok, x, 0.0), kern, mode="same")
        den = np.convolve(ok.astype(float), kern, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            xm = np.where(den > 0.999, num / np.where(den > 0, den, 1.0), np.nan)
        g = -(xm[1:] - xm[:-1]) / (zs[1:] - zs[:-1])
        if not np.isfinite(g).any():
            continue
        k = int(np.nanargmax(g))
        out[n] = 0.5 * (zs[k] + zs[k + 1])
    return out


def salinity_on_theta(z: np.ndarray, th: np.ndarray, S: np.ndarray, theta: float = 6.0,
                      z_min: float = 150.0) -> np.ndarray:
    """Salinity on a potential-temperature surface (n_prof,) of binned profiles, using only bins deeper than
    ``z_min``: below the seasonal layer the θ–S relation of a water mass is stable over months, so a trend of S(θ)
    along a deployment is sensor drift (the θ–S check of Argo delayed-mode QC, Owens & Wong 2009,
    doi:10.1016/j.dsr.2008.09.008).  NaN when the deep profile does not span ``theta``."""
    th = np.atleast_2d(np.asarray(th, float))
    S = np.atleast_2d(np.asarray(S, float))
    out = np.full(th.shape[0], np.nan)
    deep = np.asarray(z) > z_min
    for n in range(th.shape[0]):
        ok = deep & np.isfinite(th[n]) & np.isfinite(S[n])
        if ok.sum() < 3:
            continue
        x, s = th[n, ok], S[n, ok]
        if not (x.min() <= theta <= x.max()):
            continue
        o = np.argsort(x, kind="stable")
        out[n] = float(np.interp(theta, x[o], s[o]))
    return out


def pair_profiles(ta: np.ndarray, lata: np.ndarray, lona: np.ndarray, tb: np.ndarray, latb: np.ndarray,
                  lonb: np.ndarray, max_km: float = 5.0,
                  max_dt: np.timedelta64 = np.timedelta64(2, "h")) -> tuple[np.ndarray, np.ndarray]:
    """All pairs (i of set a, j of set b) closer than ``max_km`` and ``max_dt``; returns (ia, ib) int arrays."""
    ta = np.asarray(ta, "datetime64[ns]")
    tb = np.asarray(tb, "datetime64[ns]")
    order = np.argsort(tb)
    tbs = tb[order]
    ia, ib = [], []
    mdt = np.timedelta64(max_dt, "ns")
    for i in range(ta.size):
        lo, hi = np.searchsorted(tbs, ta[i] - mdt), np.searchsorted(tbs, ta[i] + mdt, side="right")
        if hi <= lo:
            continue
        j = order[lo:hi]
        d = haversine_km(lata[i], lona[i], latb[j], lonb[j])
        j = j[d <= max_km]
        ia += [i] * j.size
        ib += list(j)
    return np.array(ia, np.int64), np.array(ib, np.int64)


def bias_drift(t: np.ndarray, d: np.ndarray, max_points: int = 2000) -> dict:
    """Robust bias and drift of differences d (platform − reference) at times t: median, σ = 1.4826 MAD, n, and the
    Theil–Sen slope (Sen 1968) expressed per 30 days (subsampled evenly to ``max_points`` for the O(n²) slope)."""
    from scipy.stats import theilslopes

    t = np.asarray(t, "datetime64[ns]")
    d = np.asarray(d, float)
    ok = np.isfinite(d)
    t, d = t[ok], d[ok]
    if d.size == 0:
        return dict(n=0, median=np.nan, sigma=np.nan, drift_per_30d=np.nan)
    med = float(np.median(d))
    sig = float(1.4826 * np.median(np.abs(d - med)))
    days = (t - t.min()) / np.timedelta64(1, "D")
    slope = np.nan
    if d.size >= 10 and np.ptp(days) > 2.0:
        sel = np.linspace(0, d.size - 1, min(d.size, max_points)).astype(int)
        slope = float(theilslopes(d[sel], days[sel])[0])
    return dict(n=int(d.size), median=med, sigma=sig, drift_per_30d=30.0 * slope)


def classify(accepted: float, problems: list[str], fatal: list[str] | None = None, partial_below: float = 0.95,
             corrupted_below: float = 0.5) -> str:
    """Verdict of a record: 'corrupted' if any ``fatal`` problem or less than ``corrupted_below`` of its samples
    pass; 'partial' if less than ``partial_below`` pass or any non-fatal ``problems`` (gaps, excluded intervals);
    'usable' otherwise."""
    if fatal or accepted < corrupted_below:
        return "corrupted"
    if accepted < partial_below or problems:
        return "partial"
    return "usable"
