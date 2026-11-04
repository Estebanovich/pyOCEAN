#!/usr/bin/env python
"""
Model-vs-observation comparison of the NWA12 cut (regional chain plan Task 11; spec 2026-11-01-regional-chain,
criteria 4 and 7): pyOCEAN and MOM6 on the same cut against the QC'd observations of ``obs_qc.py``.

Only samples that the observation QC accepts enter a metric (``figs/obs_qc/coops_qc.json`` excluded intervals:
gaps, CO-OPS-inferred samples, spikes, stuck values, gross errors and confirmed steps; rejected step candidates and
F/R/L-flagged samples are kept, as the QC decided).  Sections:

1. **Tides only (criterion 4).**  Least-squares harmonic analysis (``diagnostics.sealevel_qc.harmonic_analysis``,
   OTPS basis with nodal corrections, the convention of ``forcing.tides``) of the hourly η of the 30-day tides-only
   runs at each gauge's cell, after the first day; M2, S2, N2, K1, O1, Q1 solved, K2 and P1 inferred from S2 and K1
   with the CO-OPS ratios (a 29-day record separates neither pair: Rayleigh R = 1).  Truth: the CO-OPS ``harcon``
   constants (5-yr vector mean 2018–2022; the QC found the 2017 records within 0.2–2.2 cm of them in M2).  TPXO10 at
   the nearest wet atlas node to the gauge as a third column; Woods Hole, Nantucket, Montauk and The Battery are
   reported apart (the QC found TPXO10 not representative there, class d).  Vector error |h_mod − h_obs|.
2. **Full run, η at the gauges (first look, no score).**  Verified hourly heights, each series' own mean removed
   (the gauges sit 16–20 cm above the 1983–2001 MSL datum); total η, the tide of the full run (same analysis, 122
   days), and the subtidal η (Godin 24-24-25 h filter) raw and after removing the inverse-barometer response
   η_IB = −(p_a − ⟨p_a⟩)/(ρ₀ g) from the observation (ERA5 ``msl`` at the nearest node; neither model has
   atmospheric pressure in the parity configuration, NWA12 ``MAX_P_SURF = 0``).  Long Island Sound overtides
   (±0.3 m at Bridgeport) are not in the 8-constituent basis: there the total-η metric is the one to read.
3. **SST against GLORYS12** (daily mean of the top level, 0.49 m, bilinear to the T points): pyOCEAN has daily
   00 UTC *snapshots* of the full field (``out/``), MOM6 daily *means* (``tos``).  GLORYS is linearly interpolated to
   00 UTC for the snapshots; the snapshot-minus-daily-mean bias of pyOCEAN is measured at the 13 series points from
   the per-step series and reported next to it.
4. **NDBC 44025 SST** (``WTMP``, hourly, the QC's accepted hours): pyOCEAN per-step top-layer T at the buoy's cell
   (hourly and daily means), MOM6 daily-mean ``tos`` at the same cell, GLORYS and OISST at the buoy.  The buoy has no
   salinity and no subsurface temperature (``44025o2017`` does not exist), so S and T(z) at 44025 are not compared.
5. **SST against OISST v2.1** (NOAA 1/4° daily AVHRR-only analysis, Reynolds et al. 2007, doi:10.1175/2007JCLI1824.1;
   ``data/oisst/``, read with xarray).  QC first (``oisst_qc``: day coverage, 12 UTC stamps, land mask, range, ice,
   analysis error, coastal representativeness against GLORYS12; bias against the QC'd 44025 daily means), then the
   §3 comparison with OISST as reference (bilinear from the accepted pixels; GLORYS12 as a fourth source) and the
   snapshot-vs-daily-mean effect measured on the hourly run ``full_dt600_bih30h`` (``oisst_snapshot_effect``).

6. **In situ** (``obs_qc_insitu.py`` accepted samples, ``figs/obs_qc/insitu_accepted.npz``): glider and OOI profiler
   profiles (θ, S bias/RMSE per depth band on common samples; MLD = σ₀ + 0.03 from 5 m, thermocline, layer means,
   the KPP-vs-ePBL test ``mld_test``), moored CTDs at their depth, eMOLT bottom T, ADCP depth-mean currents, and the
   slope / Gulf Stream water on the southern MAB inner shelf (``shelf_intrusion``).  pyOCEAN snapshots are matched
   to observations within ±30 min (±20 min for ADCP ensembles), MOM6 and GLORYS daily means to daily means.

Outputs (never in the repo): ``figs/obs_compare/`` — ``tides.csv``, ``eta_full.csv``, ``sst_glorys.csv``,
``sst_44025.csv``, ``sst_oisst.csv``, ``oisst_qc.json``, ``summary.json`` and figures.

Usage:
    python validation/nwa12_cut/compare.py [--tides-py RUN …] [--full-py RUN] [--mom6-tides RUN] [--mom6-full RUN …]
        [--hourly-py RUN] [--oisst DIR] [--insitu NPZ]
        [--only tides eta sst buoy oisst profiles moorings emolt adcp intrusion]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diagnostics.sealevel_qc import harmonic_analysis  # noqa: E402

RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut")
QC = RUNS / "figs" / "obs_qc"
OUT = RUNS / "figs" / "obs_compare"
T0 = np.datetime64("2017-05-01T00:00", "ns")
HOUR = np.timedelta64(1, "h")
CONS = ["m2", "s2", "n2", "k1", "o1", "q1", "k2", "p1"]      # = obs_qc.CONS
REPORT = ["m2", "s2", "n2", "k1", "o1"]                       # plan Task 11
INFER = {"k2": "s2", "p1": "k1"}
TPXO_POOR = {"8447930": "Woods Hole", "8449130": "Nantucket", "8510560": "Montauk", "8518750": "The Battery"}
#: QC categories that remove a sample (obs_qc README: rejected step candidates and F/R/L flags are kept)
EXCLUDE = ("gaps", "inferred", "spikes", "stuck", "gross", "steps")
SECTIONS = ("tides", "eta", "sst", "buoy", "oisst", "profiles", "moorings", "emolt", "adcp", "intrusion")
RHO0, G = 1035.0, 9.8                                          # parity ρ₀ and g (R21)
GODIN_HALF = 35                                                # 24-24-25 h running means: support ±35 h


# ------------------------------------------------------------------------------------------------ pure helpers
def tidal_constants(t: np.ndarray, h: np.ndarray, constituents: list[str],
                    inference: dict[str, tuple[str, complex]] | None = None) -> dict:
    """Harmonic constants of a sea-level series in the OTPS convention (``harmonic_analysis``).

    h(t) = h̄ + Σ_c f_c [Re h_c cos θ_c − Im h_c sin θ_c]  (``forcing.tides.predict`` basis), h_c = A_c e^{−i G_c}
    with A the amplitude [m] and G the Greenwich phase lag [°] (the CO-OPS ``phase_GMT`` convention).

    Returns dict(hc complex (nc,), amp [m], phase [° in 0–360), mean [m], residual [m]).
    """
    r = harmonic_analysis(t, h, constituents, inference=inference)
    hc = r["hc"]
    return dict(hc=hc, amp=np.abs(hc), phase=np.rad2deg(-np.angle(hc)) % 360.0, mean=r["mean"],
                residual=r["residual"])


def tidal_errors(h_mod: np.ndarray, h_obs: np.ndarray) -> dict:
    """Amplitude ratio A_mod/A_obs, phase difference G_mod − G_obs [°, wrapped to ±180; > 0: model lags] and vector
    error |h_mod − h_obs| [m] of complex OTPS constants."""
    h_mod, h_obs = np.asarray(h_mod, complex), np.asarray(h_obs, complex)
    dphase = (np.rad2deg(np.angle(h_obs) - np.angle(h_mod)) + 180.0) % 360.0 - 180.0
    return dict(amp_ratio=np.abs(h_mod) / np.abs(h_obs), dphase=dphase, vector=np.abs(h_mod - h_obs))


def undo_step_average(hc: np.ndarray, constituents: list[str], n: int, dt: float) -> np.ndarray:
    """Instantaneous constants from those of a series whose samples are the mean of the ``n`` step-end values
    x(t − kΔt), k = 0…n−1 (MOM6's SSH diagnostic: the time mean over the coupling cycle, ``ssh_rint``/``time_in_cycle``,
    MOM6 0e9a561 src/core/MOM.F90:1056, 1100, 1146).  For x = A cos(ωt − G) that mean is
    A g cos(ωt − G − ω(n−1)Δt/2) with g = sin(nωΔt/2)/(n sin(ωΔt/2)), so h = h_meas e^{iω(n−1)Δt/2}/g."""
    from forcing.tides import OMEGA

    w = np.array([OMEGA[c] for c in constituents])
    g = np.sin(n * w * dt / 2) / (n * np.sin(w * dt / 2))
    return np.asarray(hc, complex) * np.exp(1j * w * (n - 1) * dt / 2) / g


def accepted(t: np.ndarray, excluded: dict[str, list]) -> np.ndarray:
    """True where a sample at times ``t`` is accepted by the QC: outside every interval [start, end] (inclusive,
    'YYYY-MM-DDTHH:MM' strings) of the excluding categories ``EXCLUDE`` of ``coops_qc.json``."""
    t = np.asarray(t, "datetime64[ns]")
    ok = np.ones(t.size, bool)
    for cat in EXCLUDE:
        for iv in excluded.get(cat, []):
            a = np.datetime64(iv[0], "ns")
            b = np.datetime64(iv[-1], "ns")
            ok &= ~((t >= a) & (t <= b))
    return ok


def _running_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Centred running mean over n samples (n even: the window ends half a sample late, as Godin's 24 h), NaN
    wherever the window is incomplete or holds a NaN."""
    out = np.full(x.size, np.nan)
    if x.size < n:
        return out
    bad = ~np.isfinite(x)
    c = np.concatenate([[0.0], np.cumsum(np.where(bad, 0.0, x))])
    cb = np.concatenate([[0], np.cumsum(bad)])
    s = (c[n:] - c[:-n]) / n
    nb = cb[n:] - cb[:-n]
    lo = (n - 1) // 2
    out[lo:lo + s.size] = np.where(nb == 0, s, np.nan)
    return out


def godin(x: np.ndarray) -> np.ndarray:
    """Godin (1972) low-pass of an hourly series: running means of 24, 24 and 25 h (A24² A25 / 24² 25), which
    removes the diurnal and semidiurnal tides (M2 amplitude response ≈ 3e-3); support ±35 h, NaN where incomplete."""
    y = _running_mean(np.asarray(x, float), 24)
    y = _running_mean(y[::-1], 24)[::-1]          # the second 24-h mean ends half an hour early: centred pair
    return _running_mean(y, 25)


def daily_mean(t: np.ndarray, x: np.ndarray, min_count: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """UTC daily means of a series or of a (nt, ...) field: (day starts datetime64[D], mean (ndays, ...)), NaN where
    fewer than ``min_count`` finite samples fall in the day (counted per element of a field)."""
    d = np.asarray(t, "datetime64[ns]").astype("datetime64[D]")
    x = np.asarray(x, float)
    days = np.unique(d)
    out = np.full((days.size,) + x.shape[1:], np.nan)
    for k, day in enumerate(days):
        v = x[d == day]
        ok = np.isfinite(v)
        n = ok.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(ok, v, 0.0).sum(axis=0) / n
        out[k] = np.where(n >= min_count, m, np.nan)
    return days, out


def skill(obs: np.ndarray, mod: np.ndarray) -> dict:
    """n, bias (mod − obs), RMSE, centred RMSE, correlation and std ratio σ_mod/σ_obs on the samples where both are
    finite."""
    obs, mod = np.asarray(obs, float), np.asarray(mod, float)
    ok = np.isfinite(obs) & np.isfinite(mod)
    o, m = obs[ok], mod[ok]
    if o.size < 3:
        return dict(n=int(o.size), bias=np.nan, rmse=np.nan, crmse=np.nan, corr=np.nan, std_ratio=np.nan)
    d = m - o
    so, sm = o.std(), m.std()
    corr = float(np.corrcoef(o, m)[0, 1]) if so > 0 and sm > 0 else (1.0 if np.allclose(d, d.mean()) else np.nan)
    return dict(n=int(o.size), bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())),
                crmse=float(np.sqrt(((d - d.mean()) ** 2).mean())), corr=corr,
                std_ratio=float(sm / so) if so > 0 else np.nan)


def inverse_barometer(p: np.ndarray, rho0: float = RHO0, g: float = G) -> np.ndarray:
    """Static inverse-barometer sea level η_IB = −(p_a − ⟨p_a⟩)/(ρ₀ g) [m] of a pressure series p_a [Pa]
    (Wunsch & Stammer 1997, eq. 1), the mean taken over the finite samples."""
    p = np.asarray(p, float)
    return -(p - np.nanmean(p)) / (rho0 * g)


# ------------------------------------------------------------------------------------------------ loading
def qc_table(qc: Path = QC) -> tuple[dict, dict]:
    """(rows by station id, excluded intervals by station id) of ``coops_qc.json``."""
    j = json.loads((qc / "coops_qc.json").read_text())
    return {r["id"]: r for r in j["rows"]}, j["excluded"]


def obs_hourly(sid: str, excluded: dict, tides: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Verified hourly heights [m, MSL] of one gauge with the QC-rejected hours set to NaN, on the regular hourly
    grid of the period (missing hours NaN)."""
    from validation.nwa12_cut.obs_qc import TIDES, load_coops

    _, df = load_coops(tides or TIDES, sid, "wlh")
    t = df["t"].to_numpy("datetime64[ns]")
    v = df["v"].to_numpy(float)
    v = np.where(accepted(t, excluded), v, np.nan)
    grid = np.arange(T0, np.datetime64("2017-09-01T00:00", "ns") + HOUR, HOUR)
    out = np.full(grid.size, np.nan)
    k = ((t - T0) // HOUR).astype(int)
    on = ((t - T0) % HOUR == np.timedelta64(0)) & (k >= 0) & (k < grid.size)
    out[k[on]] = v[on]
    return grid, out


def harcon_ratio(h: np.ndarray) -> dict[str, tuple[str, complex]]:
    """Inference {K2: (S2, r), P1: (K1, r)} from a gauge's CO-OPS constants (ordered as CONS)."""
    pos = {c: k for k, c in enumerate(CONS)}
    return {c: (ref, complex(h[pos[c]] / h[pos[ref]])) for c, ref in INFER.items()}


def step_average_of(run: Path) -> tuple[int, float]:
    """(n_avg, Δt [s]) of a run's η samples: a MOM6 run (``*ocean_hourly*.nc``) writes the mean of the
    n_avg = dt_cpld/DT step-end values of each coupling hour (MOM6 0e9a561 src/core/MOM.F90:1050-1100, ``ssh_rint``
    / ``time_in_cycle``), a pyOCEAN run instantaneous values (n_avg = 1, Δt = the sample spacing of ``series.npz``)."""
    import re

    run = Path(run)
    if any(run.glob("*ocean_hourly*.nc")):
        cpl = float(re.search(r"dt_cpld\s*=\s*([0-9.]+)", (run / "input.nml").read_text()).group(1))
        dt = float(re.search(r"^DT\s*=\s*([0-9.]+)", (run / "MOM_parameter_doc.short").read_text(), re.M).group(1))
        return int(round(cpl / dt)), dt
    return 1, float(np.median(np.diff(np.load(run / "series.npz")["t"])))


def model_eta(run: Path) -> tuple[np.ndarray, list[str], np.ndarray, int, float]:
    """(hourly times, station names, η (nt, n), n_avg, Δt [s]) of a run: pyOCEAN ``series.npz`` (instantaneous
    step-end values, the on-the-hour ones kept; n_avg = 1) or MOM6 ``*ocean_hourly*.nc`` SSH at the same cells, which
    is the mean of the n_avg = dt_cpld/DT step-end values of the coupling hour (``step_average_of``,
    ``undo_step_average``)."""
    run = Path(run)
    n_avg, dt = step_average_of(run)
    if n_avg > 1:
        from validation.nwa12_cut.check_mom6 import mom6_eta
        sec, names, eta = mom6_eta(run)
    else:
        s = np.load(run / "series.npz")
        sec, names, eta = s["t"], [str(x) for x in s["names"]], s["eta"]
    on = np.isclose(sec % 3600.0, 0.0)
    t = T0 + (np.round(sec[on]) * 1e9).astype("timedelta64[ns]")
    return t, names, eta[on], n_avg, dt


def model_constants(t: np.ndarray, x: np.ndarray, n_avg: int, dt: float, inference: dict) -> np.ndarray:
    """Instantaneous OTPS constants (CONS) of a model series, the step average undone when n_avg > 1."""
    hc = tidal_constants(t, x, CONS, inference=inference)["hc"]
    return undo_step_average(hc, CONS, n_avg, dt) if n_avg > 1 else hc


def obs_6min(sid: str, excluded: dict, tides: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Verified 6-min heights [m, MSL] of one gauge, QC-rejected samples NaN."""
    from validation.nwa12_cut.obs_qc import TIDES, load_coops

    _, df = load_coops(tides or TIDES, sid, "wl")
    t = df["t"].to_numpy("datetime64[ns]")
    v = df["v"].to_numpy(float)
    return t, np.where(accepted(t, excluded), v, np.nan)


def box_mean(t_src: np.ndarray, x: np.ndarray, t: np.ndarray, n: int, dt: float) -> np.ndarray:
    """Mean of x interpolated (linear) to t − kΔt, k = 0…n−1: the observation seen through MOM6's step average.
    NaN when any of the n values falls next to a missing sample."""
    xs = (t_src - T0) / np.timedelta64(1, "s")
    acc = np.zeros(t.size)
    for k in range(n):
        tk = (t - T0) / np.timedelta64(1, "s") - k * dt
        acc += np.interp(tk, xs, x, left=np.nan, right=np.nan)
    return acc / n


def era5_msl(lat: float, lon: float, t: np.ndarray) -> np.ndarray:
    """ERA5 mean sea-level pressure [Pa] at the nearest 0.25° node, at hourly times ``t``."""
    import xarray as xr

    files = sorted((RUNS / "data" / "era5").glob("era5_*_instant.nc"))
    with xr.open_mfdataset(files, combine="by_coords") as e:
        p = e["msl"].sel(latitude=lat, longitude=lon, method="nearest").sel(valid_time=t, method="nearest")
        return p.values.astype(float)


# ------------------------------------------------------------------------------------------------ 1. tides
def tides_compare(runs: dict[str, Path], qc: Path = QC, skip_days: float = 1.0, tpxo: bool = True) -> list[dict]:
    """One row per (gauge, source, constituent): amplitude, phase, ratio, phase difference and vector error against
    the CO-OPS harcon constants.  ``runs`` = {label: run directory}; TPXO10 added as label 'TPXO10'."""
    from validation.nwa12_cut.obs_qc import TIDES, TPXO, harcon, tpxo_at

    rows_qc, _ = qc_table(qc)
    rows = []
    series = {lab: model_eta(r) for lab, r in runs.items()}
    for sid, q in rows_qc.items():
        h_obs = harcon(TIDES, sid)
        inf = harcon_ratio(h_obs)
        src = {}
        for lab, (t, names, eta, n_avg, dt) in series.items():
            if f"coops_{sid}" not in names:
                continue
            keep = t >= T0 + np.timedelta64(int(skip_days * 86400), "s")
            src[lab] = model_constants(t[keep], eta[keep, names.index(f"coops_{sid}")], n_avg, dt, inf)
        if tpxo:
            src["TPXO10"], _ = tpxo_at(TPXO, q["lat"], q["lon"])
        for lab, h in src.items():
            e = tidal_errors(h, h_obs)
            for k, c in enumerate(CONS):
                if c not in REPORT:
                    continue
                rows.append(dict(id=sid, name=q["name"], verdict=q["verdict"], source=lab, constituent=c.upper(),
                                 amp_obs=float(abs(h_obs[k])), amp=float(abs(h[k])),
                                 phase_obs=float(np.rad2deg(-np.angle(h_obs[k])) % 360),
                                 phase=float(np.rad2deg(-np.angle(h[k])) % 360), amp_ratio=float(e["amp_ratio"][k]),
                                 dphase=float(e["dphase"][k]), vector=float(e["vector"][k]),
                                 tpxo_poor=sid in TPXO_POOR))
    return rows


def tides_summary(rows: list[dict]) -> dict:
    """Median and mean M2 vector error per source over the gauges every source has, all and without TPXO_POOR,
    plus the RSS of the five constituents (√(½ Σ|Δh|²))."""
    srcs = sorted({r["source"] for r in rows})
    common = set.intersection(*[{r["id"] for r in rows if r["source"] == s} for s in srcs])
    out = {}
    for s in srcs:
        for subset, ids in (("all", common), ("tpxo_good", common - set(TPXO_POOR))):
            m2 = [r["vector"] for r in rows if r["source"] == s and r["constituent"] == "M2" and r["id"] in ids]
            rss = [np.sqrt(0.5 * sum(r["vector"] ** 2 for r in rows if r["source"] == s and r["id"] == i))
                   for i in sorted(ids)]
            out[f"{s}|{subset}"] = dict(n=len(m2), m2_median=float(np.median(m2)), m2_mean=float(np.mean(m2)),
                                        rss_median=float(np.median(rss)), rss_mean=float(np.mean(rss)))
    return out


# ------------------------------------------------------------------------------------------------ 2. η, full run
def eta_compare(runs: dict[str, Path], qc: Path = QC, skip_days: float = 1.0) -> tuple[list[dict], dict]:
    """η at the gauges against the QC'd hourly heights, per run: total (demeaned), subtidal raw and IB-corrected
    (Godin), and the M2 vector error of the full run's own tide.  Returns (rows, series for figures)."""
    from validation.nwa12_cut.obs_qc import TIDES, harcon

    rows_qc, excl = qc_table(qc)
    series = {lab: model_eta(r) for lab, r in runs.items()}
    rows, figs = [], {}
    t_start = T0 + np.timedelta64(int(skip_days * 86400), "s")
    for sid, q in rows_qc.items():
        t, o_h = obs_hourly(sid, excl[sid])
        t6, v6 = obs_6min(sid, excl[sid])
        pa = era5_msl(q["lat"], q["lon"], t)
        ib = inverse_barometer(pa)
        h_obs = harcon(TIDES, sid)
        inf = harcon_ratio(h_obs)
        keep_t = t >= t_start
        figs[sid] = dict(t=t, obs=o_h, ib=ib, mod={})
        for lab, (tm, names, eta, n_avg, dt) in series.items():
            if f"coops_{sid}" not in names:
                continue
            # the observation seen as the model reports it: on-the-hour value, or MOM6's step average
            o = o_h if n_avg == 1 else box_mean(t6, v6, t, n_avg, dt)
            m = np.full(t.size, np.nan)
            k = ((tm - T0) // HOUR).astype(int)
            ok = (k >= 0) & (k < t.size)
            m[k[ok]] = eta[ok, names.index(f"coops_{sid}")]
            both = keep_t & np.isfinite(o) & np.isfinite(m)
            oo = np.where(both, o - np.nanmean(o[both]), np.nan)
            mm = np.where(both, m - np.nanmean(m[both]), np.nan)
            # subtidal: filter the full model series (no gaps) and the observation (NaN-aware), then keep both
            ms = godin(np.where(keep_t, m, np.nan))
            os_ = godin(np.where(keep_t, o, np.nan))
            osi = godin(np.where(keep_t, o - ib, np.nan))
            sb = np.isfinite(ms) & np.isfinite(os_)
            ms_, os2, osi2 = (x - np.nanmean(x[sb]) if sb.any() else x for x in (ms, os_, osi))
            ms_, os2, osi2 = (np.where(sb, x, np.nan) for x in (ms_, os2, osi2))
            tc = model_constants(tm[tm >= t_start], eta[tm >= t_start, names.index(f"coops_{sid}")], n_avg, dt,
                                 inf)
            e = tidal_errors(tc, h_obs)
            tot, sub, subib = skill(oo, mm), skill(os2, ms_), skill(osi2, ms_)
            rows.append(dict(id=sid, name=q["name"], verdict=q["verdict"], source=lab, n_hours=tot["n"],
                             total_rmse=tot["rmse"], total_corr=tot["corr"], total_std_ratio=tot["std_ratio"],
                             n_subtidal=sub["n"], sub_rmse=sub["rmse"], sub_corr=sub["corr"],
                             sub_std_ratio=sub["std_ratio"], sub_ib_rmse=subib["rmse"], sub_ib_corr=subib["corr"],
                             sub_ib_std_ratio=subib["std_ratio"], obs_sub_std=float(np.nanstd(os2)),
                             obs_sub_ib_std=float(np.nanstd(osi2)), m2_vector=float(e["vector"][0]),
                             m2_amp_ratio=float(e["amp_ratio"][0]), m2_dphase=float(e["dphase"][0])))
            figs[sid]["mod"][lab] = dict(total=mm, sub=ms_, obs_sub=os2, obs_sub_ib=osi2)
    return rows, figs


# ------------------------------------------------------------------------------------------------ 3. SST vs GLORYS
def _glorys_sst(lat2: np.ndarray, lon2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(daily times at 12 UTC, GLORYS12 top-level θ (nt, ny, nx) bilinear at the model T points)."""
    import xarray as xr

    from validation.nwa12_cut.obs_qc import GLORYS

    with xr.open_dataset(GLORYS) as g:
        s = g["thetao"].isel(depth=0).sel(time=slice("2017-04-30", "2017-09-02"))
        si = s.interp(latitude=xr.DataArray(lat2, dims=("y", "x")), longitude=xr.DataArray(lon2, dims=("y", "x")))
        return s["time"].values.astype("datetime64[ns]") + 12 * HOUR, si.values


def _pyocean_sst(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """(times, top-layer T (nt, ny, nx)) of the daily snapshots of a pyOCEAN run (``out/out_*.nc``)."""
    import netCDF4

    fs = sorted(p for p in (Path(run) / "out").glob("out_*.nc") if not p.name.startswith("._"))
    ts, sst = [], []
    for p in fs:
        stamp = p.stem[4:]
        ts.append(np.datetime64(f"{stamp[:10]}T{stamp[11:13]}:{stamp[13:15]}", "ns"))
        with netCDF4.Dataset(p) as ds:
            v = ds["T"]
            v.set_auto_mask(False)
            sst.append(np.asarray(v[0, -1], float))
    return np.array(ts), np.array(sst)


def _mom6_sst(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """(times = centre of each daily mean, daily-mean tos (nt, ny, nx)) of a MOM6 run."""
    import netCDF4

    from validation.nwa12_cut.check_mom6 import _find, _read

    with netCDF4.Dataset(_find(run, "ocean_daily")) as ds:
        tv = ds["time"]
        t = np.array([np.datetime64(x.isoformat(), "ns") for x in
                      netCDF4.num2date(tv[:], tv.units, getattr(tv, "calendar", "gregorian"))])
        return t, _read(ds, "tos")


def _interp_time(tg: np.ndarray, g: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation in time of a (nt, ny, nx) field to the times ``t`` (NaN outside)."""
    x = (tg - tg[0]) / HOUR
    y = (t - tg[0]) / HOUR
    out = np.full((t.size,) + g.shape[1:], np.nan)
    for k, yy in enumerate(y):
        i = np.searchsorted(x, yy)
        if 0 < i < x.size:
            w = (yy - x[i - 1]) / (x[i] - x[i - 1])
            out[k] = (1 - w) * g[i - 1] + w * g[i]
        elif i < x.size and x[i] == yy:
            out[k] = g[i]
    return out


def region_skill(t: np.ndarray, f: np.ndarray, ref: np.ndarray, regions: dict[str, np.ndarray], area: np.ndarray,
                 periods: dict[str, np.ndarray], source: str = "") -> tuple[list[dict], dict]:
    """Area-weighted SST skill of a model field against a reference on the same grid and times.

    For each region mask R and day n: bias_n = Σ_R A (f − r) / Σ_R A and ms_n = Σ_R A (f − r)² / Σ_R A [°C, °C²]
    (A = cell area [m²]; NaN cells count as zero difference, so the masks must hold only cells where both are
    defined).  A period's bias is the mean of bias_n and its RMSE √(mean ms_n).

    Returns (rows: dicts source/region/period/n_days/bias/rmse/cells, bias_t: {region: (t, bias_n)})."""
    d = np.asarray(f, float) - np.asarray(ref, float)
    rows, bias_t = [], {}
    for reg, m in regions.items():
        w = np.where(m, area, 0.0)
        dz = np.where(m, np.nan_to_num(d), 0.0)
        b_n = (dz * w).sum(axis=(1, 2)) / w.sum()
        ms_n = (dz ** 2 * w).sum(axis=(1, 2)) / w.sum()
        bias_t[reg] = (t, b_n)
        for per, sel in periods.items():
            rows.append(dict(source=source, region=reg, period=per, n_days=int(sel.sum()),
                             bias=float(b_n[sel].mean()), rmse=float(np.sqrt(ms_n[sel].mean())), cells=int(m.sum())))
    return rows, bias_t


def sst_regions(H: np.ndarray, ref_ok: np.ndarray, edge: int = 10, h_shelf: float = 200.0) -> dict[str, np.ndarray]:
    """Comparison regions on the model grid: all wet cells where the reference is defined every day, and the interior
    (≥ ``edge`` cells from the open edges) split into shelf (H < ``h_shelf`` m) and deep water."""
    wet = np.isfinite(H) & ref_ok
    ny, nx = H.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    interior = (jj >= edge) & (jj < ny - edge) & (ii >= edge) & (ii < nx - edge)
    return {"all": wet, "interior_shelf": wet & interior & (H < h_shelf),
            "interior_deep": wet & interior & (H >= h_shelf)}


def _periods(t: np.ndarray) -> dict[str, np.ndarray]:
    jun = np.datetime64("2017-06-01", "ns")
    return {"May": t < jun, "Jun-Aug": t >= jun, "all": np.ones(t.size, bool)}


def sst_compare(py_run: Path, mom6_runs: dict[str, Path], ref: tuple[np.ndarray, np.ndarray] | None = None,
                extra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> tuple[list[dict], dict]:
    """Daily domain SST metrics against a reference (default GLORYS12; OISST with ``ref=oisst_on_model(...)``) on
    the common wet cells where the reference is defined every day, for the whole cut and its interior split into shelf
    and deep water (``sst_regions``).  Each source is compared at its own times (pyOCEAN 00 UTC snapshots, MOM6 and
    GLORYS daily means centred at 12 UTC) with the reference interpolated linearly in time.  ``extra`` adds sources
    given as (times, (nt, ny, nx) field), e.g. GLORYS against OISST."""
    from validation.nwa12_cut import case as cs

    hg, H = cs.grid(cs.CASE)
    lat2, lon2 = hg["lat_t"].values, hg["lon_t"].values
    area = hg["area_t"].values
    tg, gl = ref if ref is not None else _glorys_sst(lat2, lon2)
    src = {"pyOCEAN": _pyocean_sst(py_run)}
    src.update({lab: _mom6_sst(r) for lab, r in mom6_runs.items()})
    src.update(extra or {})
    regions = sst_regions(H, np.all(np.isfinite(gl), axis=0))
    wet = regions["all"]
    rows, maps = [], {}
    t_lo = T0 + np.timedelta64(1, "D")
    for lab, (t, f) in src.items():
        keep = (t >= t_lo) & (t <= np.datetime64("2017-09-01T00:00", "ns"))
        t, f = t[keep], f[keep]
        g = _interp_time(tg, gl, t)
        ok = np.isfinite(g[:, wet]).all(axis=1)                    # reference covers the day
        t, f, g = t[ok], f[ok], g[ok]
        r_, bias_t = region_skill(t, f, g, regions, area, _periods(t), source=lab)
        rows += r_
        jja = t >= np.datetime64("2017-06-01", "ns")
        maps[lab] = dict(bias_t=bias_t, mean_bias_jja=np.where(wet, np.nanmean(f[jja] - g[jja], axis=0), np.nan))
    maps["_grid"] = dict(lat=lat2, lon=lon2, H=H)
    return rows, maps


def snapshot_bias(py_run: Path) -> dict:
    """pyOCEAN SST at 00 UTC minus its own UTC daily mean, from the per-step series at the 13 points (days 2–123):
    the error of using daily snapshots where the reference holds daily means."""
    s = np.load(Path(py_run) / "series.npz")
    t = T0 + (np.round(s["t"]) * 1e9).astype("timedelta64[ns]")
    out = {}
    for k, n in enumerate(map(str, s["names"])):
        days, dm = daily_mean(t, s["sst"][:, k], min_count=100)
        snap = s["sst"][np.isclose(s["t"] % 86400.0, 0.0), k]
        tsnap = t[np.isclose(s["t"] % 86400.0, 0.0)].astype("datetime64[D]")
        d = [snap[i] - dm[np.searchsorted(days, tsnap[i])] for i in range(1, tsnap.size - 1)
             if np.searchsorted(days, tsnap[i]) < days.size and np.isfinite(dm[np.searchsorted(days, tsnap[i])])]
        out[n] = dict(mean=float(np.mean(d)), rms=float(np.sqrt(np.mean(np.square(d)))))
    return out


# ------------------------------------------------------------------------------------------------ 5. OISST v2.1
OISST = RUNS / "data" / "oisst"
OISST_BOX = (35.5, 42.5, -76.5, -68.0)            # lat_lo, lat_hi, lon_lo, lon_hi: the cut plus one 1/4° margin
OISST_RANGE = (-1.8, 35.0)                         # physical SST range [°C] (freezing point of sea water, warm limit)
ANOM_PERSIST = 5.0                                 # |anomaly| [°C] counted per pixel for the QC report


def oisst_files(directory: Path = OISST) -> list[Path]:
    """The daily OISST v2.1 AVHRR files of a directory, sorted by date; macOS AppleDouble companions (``._*``) and
    other files are skipped."""
    return sorted(p for p in Path(directory).glob("oisst-avhrr-v02r01.*.nc") if not p.name.startswith("._"))


def coastal_pixels(land: np.ndarray) -> np.ndarray:
    """Water pixels with at least one land pixel among their 8 neighbours (outside the array counts as water)."""
    land = np.asarray(land, bool)
    p = np.pad(land, 1, constant_values=False)
    ny, nx = land.shape
    nb = np.zeros_like(land)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            if dj or di:
                nb |= p[1 + dj:1 + dj + ny, 1 + di:1 + di + nx]
    return nb & ~land


def oisst_qc(t: np.ndarray, sst: np.ndarray, err: np.ndarray, ice: np.ndarray, anom: np.ndarray,
             ref: np.ndarray | None = None, t_start: np.datetime64 = np.datetime64("2017-05-01"),
             t_end: np.datetime64 = np.datetime64("2017-09-01"), sst_range: tuple[float, float] = OISST_RANGE,
             dep_max: float = 3.0) -> dict:
    """QC of a daily OISST v2.1 record over a box (arrays (nt, ny, nx) on the 1/4° grid).

    Checks (Reynolds et al. 2007; Huang et al. 2021, OISST v2.1): one file per day from ``t_start`` to ``t_end``
    (missing days, duplicates), stamps at 12 UTC (the daily-mean convention), a constant land mask, SST inside
    ``sst_range``, no sea ice (``ice`` > 0), and the analysis error ``err`` (the OI error standard deviation [°C]).
    Representativeness: a coastal 1/4° pixel mixes land and enclosed water (AVHRR land contamination, sparse in situ
    data), so a coastal pixel whose period-mean departure from an independent reference ``ref`` (GLORYS12 on the
    same grid) exceeds ``dep_max`` [°C] is **flagged** and one without reference is **unverified**; open-ocean
    pixels are never flagged by the departure (the Gulf Stream path differs between analyses: mesoscale position,
    not an analysis fault).  Neither enters the comparison.

    Returns a dict of counts and statistics, ``pixel`` (ny, nx) in {land, ok, flagged, unverified}, ``dep`` (period-
    mean OISST − ref [°C]) and ``anom_days`` (days with |anomaly| > ``ANOM_PERSIST``)."""
    t = np.asarray(t, "datetime64[ns]")
    days = t.astype("datetime64[D]")
    want = np.arange(np.datetime64(t_start, "D"), np.datetime64(t_end, "D") + 1)
    missing = [str(d) for d in want if d not in set(days)]
    off_noon = int(np.sum((t - days.astype("datetime64[ns]")) != np.timedelta64(12, "h")))
    sst = np.asarray(sst, float)
    land_t = ~np.isfinite(sst)
    land = land_t[0]
    fin = np.isfinite(sst)
    lo, hi = sst_range
    e = np.asarray(err, float)[fin]
    ice = np.asarray(ice, float)
    with np.errstate(invalid="ignore"):
        n_ice = int(np.sum(np.isfinite(ice) & (ice > 0)))
        n_out = int(np.sum(fin & ((sst < lo) | (sst > hi))))
        anom_days = np.sum(np.abs(np.asarray(anom, float)) > ANOM_PERSIST, axis=0)
    coast = coastal_pixels(land)
    if ref is not None:
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dep = np.nanmean(sst - np.asarray(ref, float), axis=0)
    else:
        dep = np.full(land.shape, np.nan)
    pixel = np.full(land.shape, "ok", dtype="<U10")
    pixel[land] = "land"
    pixel[coast & np.isfinite(dep) & (np.abs(dep) > dep_max)] = "flagged"
    pixel[coast & ~np.isfinite(dep)] = "unverified"
    return dict(n_days=int(t.size), days_missing=missing, n_duplicates=int(t.size - np.unique(days).size),
                n_off_noon=off_noon, land_constant=bool((land_t == land).all()),
                n_land_changes=int((land_t != land).sum()), n_out_of_range=n_out, n_ice=n_ice,
                n_err_missing=int(np.sum(~np.isfinite(np.asarray(err, float)) & fin)),
                err_median=float(np.median(e)), err_p95=float(np.percentile(e, 95)), err_max=float(e.max()),
                n_water=int((~land).sum()), n_coastal=int(coast.sum()), n_flagged=int((pixel == "flagged").sum()),
                n_unverified=int((pixel == "unverified").sum()), pixel=pixel, dep=dep, anom_days=anom_days)


def load_oisst(directory: Path = OISST, box: tuple[float, float, float, float] = OISST_BOX) -> dict:
    """Read the daily OISST files over ``box`` with xarray: dict(t (12 UTC stamps), lat, lon (°E, −180…180), sst,
    err, ice, anom (nt, ny, nx) [°C, °C, %, °C], titles (set of global ``title`` attributes), files)."""
    import xarray as xr

    fs = oisst_files(directory)
    lat_lo, lat_hi, lon_lo, lon_hi = box
    parts, titles = [], set()
    for f in fs:
        with xr.open_dataset(f) as ds:
            titles.add(str(ds.attrs.get("title", "")))
            box_ = dict(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo % 360.0, lon_hi % 360.0))
            parts.append(ds.isel(zlev=0).sel(**box_).load())
    ds = xr.concat(parts, "time")
    return dict(t=ds["time"].values.astype("datetime64[ns]"), lat=ds["lat"].values.astype(float),
                lon=((ds["lon"].values.astype(float) + 180.0) % 360.0) - 180.0,
                **{v: ds[v].values.astype(float) for v in ("sst", "err", "ice", "anom")}, titles=titles,
                files=[f.name for f in fs])


def _bilinear(lat: np.ndarray, lon: np.ndarray, f: np.ndarray, lat_q: np.ndarray, lon_q: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of (nt, ny, nx) fields on a regular (lat, lon) grid to query points (any shape); NaN
    wherever one of the four corners is NaN or the point is outside the grid."""
    import xarray as xr

    da = xr.DataArray(f, dims=("time", "lat", "lon"), coords=dict(lat=lat, lon=lon))
    lat_q, lon_q = np.asarray(lat_q, float), np.asarray(lon_q, float)
    dims = tuple(f"q{k}" for k in range(lat_q.ndim))
    q = dict(lat=xr.DataArray(lat_q, dims=dims), lon=xr.DataArray(lon_q, dims=dims))
    return da.interp(**q).values


def glorys_on_oisst(o: dict) -> np.ndarray:
    """GLORYS12 top-level θ (daily means) bilinear on the OISST grid, matched by date (nt, ny, nx) [°C]."""
    import xarray as xr

    from validation.nwa12_cut.obs_qc import GLORYS

    with xr.open_dataset(GLORYS) as g:
        s = g["thetao"].isel(depth=0)
        s = s.sel(time=o["t"].astype("datetime64[D]").astype("datetime64[ns]"))
        return s.interp(latitude=xr.DataArray(o["lat"], dims="lat"),
                        longitude=xr.DataArray(o["lon"], dims="lon")).values.astype(float)


def oisst_on_model(o: dict, usable: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(times, OISST SST bilinear at the model T points (nt, ny, nx)) with the QC-rejected pixels (``usable`` False)
    removed first, so a model cell next to a flagged or unverified pixel has no reference."""
    sst = np.where(usable[None], o["sst"], np.nan)
    return o["t"], _bilinear(o["lat"], o["lon"], sst, lat2, lon2)


def oisst_qc_report(o: dict, dep_max: float = 3.0) -> tuple[dict, dict]:
    """QC of the OISST record (``oisst_qc`` against GLORYS12 on the 1/4° grid) and its bias against the QC'd NDBC
    44025 daily-mean SST at the buoy (OISST bilinear; GLORYS12 at the same point for context).  Returns (summary for
    the JSON, fields for the figure)."""
    from validation.nwa12_cut.obs_qc import NDBC_POS

    ref = glorys_on_oisst(o)
    q = oisst_qc(o["t"], o["sst"], o["err"], o["ice"], o["anom"], ref=ref, dep_max=dep_max)
    tb, xb = buoy_wtmp()
    db, mb = daily_mean(tb, xb, min_count=20)
    keep = db >= np.datetime64("2017-05-02")
    po = _bilinear(o["lat"], o["lon"], o["sst"], np.array([NDBC_POS[0]]), np.array([NDBC_POS[1]]))[:, 0]
    pe = _bilinear(o["lat"], o["lon"], o["err"], np.array([NDBC_POS[0]]), np.array([NDBC_POS[1]]))[:, 0]
    pg = _bilinear(o["lat"], o["lon"], ref, np.array([NDBC_POS[0]]), np.array([NDBC_POS[1]]))[:, 0]
    od = dict(zip(o["t"].astype("datetime64[D]"), po))
    gd = dict(zip(o["t"].astype("datetime64[D]"), pg))
    sk_o = skill(mb[keep], np.array([od.get(d, np.nan) for d in db[keep]]))
    sk_g = skill(mb[keep], np.array([gd.get(d, np.nan) for d in db[keep]]))
    flagged = [dict(lat=float(o["lat"][j]), lon=float(o["lon"][i]), status=str(q["pixel"][j, i]),
                    dep=None if not np.isfinite(q["dep"][j, i]) else float(q["dep"][j, i]),
                    anom_days=int(q["anom_days"][j, i]),
                    err_mean=float(np.nanmean(o["err"][:, j, i])))
               for j, i in np.argwhere(np.isin(q["pixel"], ("flagged", "unverified")))]
    summ = {k: v for k, v in q.items() if k not in ("pixel", "dep", "anom_days")}
    summ.update(titles=sorted(o["titles"]), first=str(o["t"][0]), last=str(o["t"][-1]), rejected_pixels=flagged,
                dep_max=dep_max, err_at_44025_mean=float(np.nanmean(pe)), buoy_daily_vs_oisst=sk_o,
                buoy_daily_vs_glorys=sk_g,
                open_ocean_dep_pct=[float(x) for x in np.nanpercentile(
                    q["dep"][(q["pixel"] == "ok") & ~coastal_pixels(q["pixel"] == "land")], [5, 50, 95])])
    fig = dict(q=q, o=o, db=db[keep], mb=mb[keep], po=po, pg=pg)
    return summ, fig


def oisst_snapshot_effect(run_hourly: Path, run_daily: Path, oisst_model: tuple[np.ndarray, np.ndarray]) -> dict:
    """Bias of comparing pyOCEAN's 00 UTC snapshots with OISST daily means, measured on an hourly-output run
    (``full_dt600_bih30h``, 30 days): the same days scored twice against OISST, once with the 00 UTC snapshots
    (reference interpolated to 00 UTC, as for ``full_dt600_bih123``) and once with UTC daily means of the 24 hourly
    fields (stamped 12 UTC).  Also checks that the hourly run's 00 UTC snapshots equal the daily run's."""
    from validation.nwa12_cut import case as cs

    hg, H = cs.grid(cs.CASE)
    area = hg["area_t"].values
    tg, go = oisst_model
    regions = sst_regions(H, np.all(np.isfinite(go), axis=0))
    th, fh = _pyocean_sst(run_hourly)
    days, dm = daily_mean(th, fh, min_count=24)
    full = np.isfinite(dm).any(axis=(1, 2))
    sel_d = full & (days >= np.datetime64("2017-05-02"))
    td = days[sel_d].astype("datetime64[ns]") + 12 * HOUR
    snap = np.isin(th, days[sel_d].astype("datetime64[ns]"))
    ts = th[snap]
    out = {}
    for lab, (t, f) in (("snapshot", (ts, fh[snap])), ("daily_mean", (td, dm[sel_d]))):
        g = _interp_time(tg, go, t)
        rows, _ = region_skill(t, f, g, regions, area, {"all": np.ones(t.size, bool)}, source=lab)
        out[lab] = {r["region"]: dict(bias=r["bias"], rmse=r["rmse"], n_days=r["n_days"], cells=r["cells"])
                    for r in rows}
    out["snapshot_minus_daily_mean"] = {reg: dict(bias=out["snapshot"][reg]["bias"] - out["daily_mean"][reg]["bias"],
                                                  rmse=out["snapshot"][reg]["rmse"] - out["daily_mean"][reg]["rmse"])
                                        for reg in regions}
    t1, f1 = _pyocean_sst(run_daily)
    common = np.intersect1d(t1, ts)
    out["identity_check_max_abs"] = float(np.nanmax(np.abs(f1[np.isin(t1, common)] - fh[np.isin(th, common)])))
    out["identity_check_days"] = int(common.size)
    out["days"] = [str(days[sel_d][0]), str(days[sel_d][-1])]
    return out


# ------------------------------------------------------------------------------------------------ 4. NDBC 44025
def buoy_wtmp() -> tuple[np.ndarray, np.ndarray]:
    """(hourly times, WTMP [°C] with the QC-rejected hours NaN): the same checks as ``obs_qc.qc_ndbc``."""
    import pandas as pd

    from diagnostics.buoy_qc import flag_reference_departure, qc_buoy_series, read_ndbc_stdmet
    from validation.nwa12_cut.obs_qc import GLORYS, NDBC_CHECKS, T1, ndbc_references

    df = read_ndbc_stdmet(RUNS / "data" / "ndbc" / "44025h2017.txt.gz")
    df = df[(df.index >= pd.Timestamp(T0) - pd.Timedelta("1h")) & (df.index < pd.Timestamp(T1))]
    minute = int(pd.Series(df.index.minute).mode().iloc[0])
    t_start = T0 - np.timedelta64(60 - minute, "m") if minute else T0
    rng, smin, tol = NDBC_CHECKS["WTMP"]
    q = qc_buoy_series(df.index.to_numpy("datetime64[ns]"), df["WTMP"].to_numpy(), rng, t_start=t_start, t_end=T1,
                       spike_min=smin)
    ref = ndbc_references(q["t"], GLORYS, RUNS / "data" / "era5")
    dep = flag_reference_departure(np.where(q["ok"], q["x"], np.nan), ref["WTMP"], min_dev=tol) & q["ok"]
    ok = q["ok"] & ~dep
    return q["t"], np.where(ok, q["x"], np.nan)


def buoy_compare(py_run: Path, mom6_runs: dict[str, Path], station: str = "ndbc_44025",
                 oisst: dict | None = None) -> tuple[list[dict], dict]:
    """44025 SST: daily means (buoy days with ≥ 20 accepted hours) of the buoy, pyOCEAN (all steps of the day), MOM6
    (daily tos at the same cell), GLORYS and OISST (bilinear at the buoy); pyOCEAN hourly against the buoy hourly; mean
    diurnal range (daily max − min) of the buoy and pyOCEAN."""
    import xarray as xr

    from validation.nwa12_cut.check_mom6 import _find, _read
    from validation.nwa12_cut.obs_qc import GLORYS, NDBC_POS

    tb, xb = buoy_wtmp()
    db, mb = daily_mean(tb, xb, min_count=20)
    s = np.load(Path(py_run) / "series.npz")
    names = [str(x) for x in s["names"]]
    k = names.index(station)
    j, i = int(s["j"][k]), int(s["i"][k])
    tp = T0 + (np.round(s["t"]) * 1e9).astype("timedelta64[ns]")
    xp = s["sst"][:, k]
    dp, mp = daily_mean(tp, xp, min_count=100)
    est = {"pyOCEAN": dict(zip(dp, mp))}
    import netCDF4
    for lab, r in mom6_runs.items():
        with netCDF4.Dataset(_find(r, "ocean_daily")) as ds:
            tv = ds["time"]
            t = np.array([np.datetime64(x.isoformat(), "ns") for x in
                          netCDF4.num2date(tv[:], tv.units, getattr(tv, "calendar", "gregorian"))])
            est[lab] = dict(zip(t.astype("datetime64[D]"), _read(ds, "tos")[:, j, i]))
    with xr.open_dataset(GLORYS) as g:
        gs = g["thetao"].isel(depth=0).interp(latitude=NDBC_POS[0], longitude=NDBC_POS[1])
        est["GLORYS12"] = dict(zip(gs["time"].values.astype("datetime64[D]"), gs.values.astype(float)))
    if oisst is not None:                                             # OISST v2.1 bilinear at the buoy
        po = _bilinear(oisst["lat"], oisst["lon"], oisst["sst"], np.array([NDBC_POS[0]]), np.array([NDBC_POS[1]]))
        est["OISST"] = dict(zip(oisst["t"].astype("datetime64[D]"), po[:, 0]))
    keep = db >= np.datetime64("2017-05-02")
    rows = []
    for lab, dd in est.items():
        m = np.array([dd.get(d, np.nan) for d in db[keep]])
        sk = skill(mb[keep], m)
        rows.append(dict(source=lab, kind="daily", **sk))
    # hourly pyOCEAN (step values at the buoy's hh:50 times, linear) against the buoy
    xh = np.interp((tb - T0) / np.timedelta64(1, "s"), s["t"], xp, left=np.nan, right=np.nan)
    sel = tb >= np.datetime64("2017-05-02T00:00", "ns")
    rows.append(dict(source="pyOCEAN", kind="hourly", **skill(xb[sel], xh[sel])))
    rng = {}
    for lab, (tt, xx) in (("buoy", (tb, xb)), ("pyOCEAN", (tb, np.where(np.isfinite(xb), xh, np.nan)))):
        d = tt.astype("datetime64[D]")
        r_ = [np.nanmax(xx[d == u]) - np.nanmin(xx[d == u]) for u in np.unique(d)
              if np.isfinite(xx[d == u]).sum() >= 20]
        rng[lab] = float(np.mean(r_))
    rows.append(dict(source="diurnal_range", kind="mean daily max-min", n=0, bias=rng["pyOCEAN"] - rng["buoy"],
                     rmse=np.nan, crmse=np.nan, corr=np.nan, std_ratio=rng["pyOCEAN"] / rng["buoy"]))
    fig = dict(db=db, mb=mb, est=est, tb=tb, xb=xb, xh=xh, ranges=rng)
    return rows, fig


# ------------------------------------------------------------------------------------------------ 6. in-situ (R26 QC)
INSITU = QC / "insitu_accepted.npz"
DEPTH_BANDS = ((0.0, 10.0), (10.0, 30.0), (30.0, 60.0), (60.0, 100.0), (100.0, 200.0), (200.0, 500.0),
               (500.0, 1000.0))
H_SHELF = 200.0
MLD_REF, MLD_DSIG, MLD_TOP = 5.0, 0.03, 7.0           # de Boyer Montégut et al. (2004) σ₀ threshold, 5 m reference
H_MIN = 1e-2                                          # m: thinner (vanished) model layers are skipped


def column_centres(h_top_first: np.ndarray) -> np.ndarray:
    """Depth below the free surface [m] of the layer centres of a column given its thicknesses, surface first."""
    h = np.asarray(h_top_first, float)
    return np.cumsum(h) - 0.5 * h


def column_to_bins(h_top_first: np.ndarray, x: np.ndarray, zb: np.ndarray) -> np.ndarray:
    """A model column (layer values ``x`` with thicknesses ``h``, surface first) on depth-bin centres ``zb``: linear
    between layer centres, the top value above the first centre, the bottom value between the last centre and the
    bed, NaN below the bed.  Layers thinner than ``H_MIN`` (vanished z* layers at the bed) are skipped."""
    h = np.asarray(h_top_first, float)
    x = np.asarray(x, float)
    ok = np.isfinite(h) & (h > H_MIN) & np.isfinite(x)
    if ok.sum() == 0:
        return np.full(zb.size, np.nan)
    zc = column_centres(np.where(ok, h, 0.0))[ok]
    out = np.interp(zb, zc, x[ok])
    out[zb > np.nansum(np.where(ok, h, 0.0))] = np.nan
    return out


def bracket(times: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(k0, k1, w) such that t ≈ (1 − w) times[k0] + w times[k1] (linear interpolation in time); k0 = −1 outside."""
    x = (np.asarray(times, "datetime64[ns]") - times[0]) / HOUR
    y = (np.asarray(t, "datetime64[ns]") - times[0]) / HOUR
    k1 = np.searchsorted(x, y)
    k1c = np.clip(k1, 1, x.size - 1)
    k0 = k1c - 1
    w = (y - x[k0]) / (x[k1c] - x[k0])
    bad = (y < x[0]) | (y > x[-1])
    return np.where(bad, -1, k0), np.where(bad, -1, k1c), np.where(bad, np.nan, w)


class _Reader:
    """Daily model/reanalysis fields read one time level at a time (cache of two levels)."""

    times: np.ndarray

    def __init__(self) -> None:
        self._cache: dict[int, dict] = {}

    def get(self, k: int) -> dict:
        if k not in self._cache:
            if len(self._cache) > 2:
                self._cache.pop(min(self._cache))
            self._cache[k] = self.load(k)
        return self._cache[k]

    def load(self, k: int) -> dict:          # pragma: no cover - abstract
        raise NotImplementedError


class PyoceanReader(_Reader):
    """pyOCEAN ``out/out_*.nc`` snapshots: T, S, h_layer (bed first in the file; returned surface first), u, v."""

    def __init__(self, run: Path) -> None:
        super().__init__()
        self.files = sorted(p for p in (Path(run) / "out").glob("out_*.nc") if not p.name.startswith("._"))
        st = [p.stem[4:] for p in self.files]
        self.times = np.array([np.datetime64(f"{s[:10]}T{s[11:13]}:{s[13:15]}", "ns") for s in st])

    def load(self, k: int, uv: bool = False) -> dict:
        import netCDF4
        out = {}
        with netCDF4.Dataset(self.files[k]) as ds:
            for v, name in (("T", "T"), ("S", "S"), ("h", "h_layer")) + ((("u", "u"), ("v", "v")) if uv else ()):
                x = ds[name]
                x.set_auto_mask(False)
                out[v] = np.asarray(x[0], float)[::-1]
        return out


class Mom6Reader(_Reader):
    """MOM6 ``ocean_daily`` means (time = centre of the day): temp, salt, h (surface first), u, v."""

    def __init__(self, run: Path) -> None:
        import netCDF4

        from validation.nwa12_cut.check_mom6 import _find
        super().__init__()
        self.path = _find(run, "ocean_daily")
        with netCDF4.Dataset(self.path) as ds:
            tv = ds["time"]
            self.times = np.array([np.datetime64(x.isoformat(), "ns") for x in
                                   netCDF4.num2date(tv[:], tv.units, getattr(tv, "calendar", "gregorian"))])

    def load(self, k: int, uv: bool = False) -> dict:
        import netCDF4

        names = (("T", "temp"), ("S", "salt"), ("h", "h")) + ((("u", "u"), ("v", "v")) if uv else ())
        with netCDF4.Dataset(self.path) as ds:
            sl = dict(names)
            out = {}
            for v, n in sl.items():
                x = ds[n]
                x.set_auto_mask(False)
                a = np.asarray(x[k], float)
                for m in [x.getncattr(a_) for a_ in ("missing_value", "_FillValue") if a_ in x.ncattrs()]:
                    a = np.where(a == m, np.nan, a)
                out[v] = a
        return out


class GlorysReader(_Reader):
    """GLORYS12 daily means (time = label + 12 h) on its z levels; ``h`` are the level-cell thicknesses."""

    def __init__(self) -> None:
        import xarray as xr

        from validation.nwa12_cut.obs_qc import GLORYS
        super().__init__()
        self.ds = xr.open_dataset(GLORYS)
        self.times = self.ds["time"].values.astype("datetime64[ns]") + 12 * HOUR
        z = self.ds["depth"].values.astype(float)
        edges = np.r_[0.0, 0.5 * (z[1:] + z[:-1]), z[-1] + 0.5 * (z[-1] - z[-2])]
        self.h = np.diff(edges)
        self.lat = self.ds["latitude"].values.astype(float)
        self.lon = self.ds["longitude"].values.astype(float)

    def load(self, k: int, uv: bool = False) -> dict:
        d = self.ds.isel(time=k)
        out = dict(T=d["thetao"].values.astype(float), S=d["so"].values.astype(float))
        if uv:
            out["u"], out["v"] = d["uo"].values.astype(float), d["vo"].values.astype(float)
        out["h"] = np.where(np.isfinite(out["T"]), self.h[:, None, None], 0.0)
        return out


def nearest_wet(lat2: np.ndarray, lon2: np.ndarray, wet: np.ndarray, lat: np.ndarray,
                lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(j, i, distance [km]) of the nearest wet cell of a (curvilinear) grid to each point (KD-tree on the unit
    sphere)."""
    from scipy.spatial import cKDTree

    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)

    jw, iw = np.nonzero(wet)
    tree = cKDTree(xyz(lat2[jw, iw], lon2[jw, iw]))
    d, n = tree.query(xyz(np.asarray(lat, float), np.asarray(lon, float)))
    return jw[n], iw[n], 2 * 6371.0 * np.arcsin(np.clip(d / 2, 0, 1))


def sample_columns(reader: _Reader, t: np.ndarray, j: np.ndarray, i: np.ndarray, zb: np.ndarray, kind: str = "ts",
                   rot: tuple[np.ndarray, np.ndarray] | None = None,
                   max_gap: np.timedelta64 = np.timedelta64(36, "h")) -> tuple[np.ndarray, np.ndarray]:
    """Two fields (n, n_bins) of a reader's columns (j, i) at the times t on the depth bins zb, linear in time
    between its two bracketing levels (snapshots or daily means; NaN if they are more than ``max_gap`` apart or t
    is outside).  ``kind`` 'ts' gives (θ, S); 'uv' gives (east, north) velocity: C-grid faces averaged to the T cell
    and rotated with ``rot`` = (cos α, sin α) (``forcing.rotation.to_earth``), or taken as they are (A-grid,
    already east/north) when ``rot`` is None.  Each time level is read once."""
    from forcing.rotation import to_earth

    k0, k1, w = bracket(reader.times, t)
    n = t.size
    out = np.zeros((2, n, zb.size))
    ok = k0 >= 0
    ok[ok] = (reader.times[k1[ok]] - reader.times[k0[ok]]) <= max_gap
    out[:, ~ok] = np.nan
    uv = kind == "uv"
    for kk in np.unique(np.r_[k0[ok], k1[ok]]):
        f = reader.load(int(kk), uv=uv) if uv else reader.get(int(kk))
        for which, wt in ((k0, 1.0 - w), (k1, w)):
            for n_ in np.flatnonzero(ok & (which == kk)):
                if wt[n_] == 0.0:
                    continue
                jj, ii = j[n_], i[n_]
                h = f["h"][:, jj, ii]
                if uv and rot is not None:
                    a_ = 0.5 * (f["u"][:, jj, ii] + f["u"][:, jj, ii + 1])
                    b_ = 0.5 * (f["v"][:, jj, ii] + f["v"][:, jj + 1, ii])
                    a_, b_ = to_earth(a_, b_, rot[0][jj, ii], rot[1][jj, ii])
                elif uv:
                    a_, b_ = f["u"][:, jj, ii], f["v"][:, jj, ii]
                else:
                    a_, b_ = f["T"][:, jj, ii], f["S"][:, jj, ii]
                out[0, n_] += wt[n_] * column_to_bins(h, a_, zb)
                out[1, n_] += wt[n_] * column_to_bins(h, b_, zb)
    return out[0], out[1]


def at_depth(x: np.ndarray, zb: np.ndarray, z: np.ndarray, clamp_bottom: bool = True) -> np.ndarray:
    """Values of binned profiles (n, n_bins) at one depth per row (linear in the bins; with ``clamp_bottom`` the
    deepest finite bin where z lies below the column, e.g. a seafloor sensor deeper than the model bed)."""
    out = np.full(x.shape[0], np.nan)
    for k in range(x.shape[0]):
        ok = np.isfinite(x[k])
        if ok.sum() < 2:
            continue
        zz = z[k] if not clamp_bottom else min(z[k], zb[ok][-1])
        out[k] = np.interp(zz, zb[ok], x[k, ok], left=np.nan, right=np.nan)
    return out


def sigma0(th: np.ndarray, S: np.ndarray, zb: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """σ₀ [kg m⁻³] of binned θ, S_P (TEOS-10 via gsw, S_A from S_P at the bin pressure)."""
    import gsw
    p = gsw.p_from_z(-zb[None, :], np.asarray(lat, float)[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        SA = gsw.SA_from_SP(S, p, np.asarray(lon, float)[:, None], np.asarray(lat, float)[:, None])
        return gsw.sigma0(SA, gsw.CT_from_pt(SA, th))


def band_stats(obs: np.ndarray, mods: dict[str, np.ndarray], zb: np.ndarray, groups: dict[str, np.ndarray],
               var: str, bands=DEPTH_BANDS) -> list[dict]:
    """Bias and RMSE of each source against the observations per depth band and profile group, on the bins where
    the observation and *every* source are defined (the same samples for all)."""
    common = np.isfinite(obs)
    for m in mods.values():
        common &= np.isfinite(m)
    rows = []
    for g, gm in groups.items():
        for z0, z1 in bands:
            sel = common & gm[:, None] & ((zb >= z0) & (zb < z1))[None, :]
            n = int(sel.sum())
            for lab, m in mods.items():
                d = (m - obs)[sel]
                rows.append(dict(var=var, group=g, band=f"{z0:.0f}-{z1:.0f}", source=lab, n=n,
                                 n_prof=int(sel.any(axis=1).sum()),
                                 bias=float(d.mean()) if n else np.nan,
                                 rmse=float(np.sqrt(np.mean(d ** 2))) if n else np.nan))
    return rows


def layer_mean(x: np.ndarray, zb: np.ndarray, z0: float, z1: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Mean of binned values between z0 and z1 [m] per profile (only bins where ``mask``), NaN if < half filled."""
    sel = (zb >= z0) & (zb < z1)
    xx = np.where(mask, x, np.nan) if mask is not None else x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmean(xx[:, sel], axis=1)
    m[np.isfinite(xx[:, sel]).sum(axis=1) < 0.5 * sel.sum()] = np.nan
    return m


def load_insitu(path: Path = INSITU) -> dict:
    """The accepted samples written by ``obs_qc_insitu.py``."""
    a = np.load(path)
    return {k: a[k] for k in a.files}


def profiles_compare(py_run: Path, mom6_runs: dict[str, Path], hourly: Path | None = None, insitu: Path = INSITU,
                     t_lo: np.datetime64 = T0 + np.timedelta64(1, "D")) -> tuple[list[dict], dict]:
    """Accepted glider and profiler-mooring profiles against pyOCEAN, MOM6 (each run) and GLORYS12.

    Each profile is sampled at the nearest wet model cell (inside the cut, ≤ 1.5 cells away), linear in time between
    the bracketing pyOCEAN 00 UTC snapshots / MOM6 and GLORYS daily means, and put on the observation's 1-m bins.
    Returns (rows: θ and S bias/RMSE per depth band, region and period; MLD/thermocline/upper-layer rows), and a
    dict with the sampled arrays for figures and the MLD test."""
    from diagnostics import profile_qc as pq
    from validation.nwa12_cut import case as cs

    a = load_insitu(insitu)
    zb = a["zb"]
    hg, H = cs.grid(cs.CASE)
    lat2, lon2 = hg["lat_t"].values, hg["lon_t"].values
    b = cs.CASE.box
    t, la, lo = a["prof_t"], a["prof_lat"], a["prof_lon"]
    inbox = (lo >= b[0]) & (lo <= b[1]) & (la >= b[2]) & (la <= b[3]) & (t >= t_lo)
    wet = np.isfinite(H) & (H > 0)
    j, i, dist = nearest_wet(lat2, lon2, wet, la, lo)
    dx = np.sqrt(hg["area_t"].values[j, i]) / 1e3
    use = inbox & (dist <= 1.5 * dx)
    idx = np.flatnonzero(use)
    t, la, lo, j, i = t[idx], la[idx], lo[idx], j[idx], i[idx]
    th_o, S_o, sig_o = a["prof_th"][idx].astype(float), a["prof_S"][idx].astype(float), a["prof_sig"][idx].astype(float)
    rec = a["prof_records"][a["prof_rec"][idx]]
    gl = GlorysReader()
    gj, gi, _ = nearest_wet(*np.meshgrid(gl.lat, gl.lon, indexing="ij"),
                            np.isfinite(gl.get(0)["T"][0]), la, lo)
    readers = {"pyOCEAN": (PyoceanReader(py_run), j, i)}
    readers.update({lab: (Mom6Reader(r), j, i) for lab, r in mom6_runs.items()})
    readers["GLORYS12"] = (gl, gj, gi)
    mod = {}
    for lab, (rd, jj, ii) in readers.items():
        mod[lab] = sample_columns(rd, t, jj, ii, zb)
        print(f"  sampled {lab}: {np.isfinite(mod[lab][0]).any(axis=1).sum()} of {t.size} profiles", flush=True)
    if hourly is not None:                    # snapshot sampling error: the hourly run at the profile time vs its
        hr = PyoceanReader(hourly)            # own 00 UTC snapshots interpolated (May only)
        day0 = PyoceanReader(hourly)
        sel0 = np.isclose(((day0.times - day0.times[0]) / HOUR) % 24, 0.0)
        day0.files = [f for f, s in zip(day0.files, sel0) if s]
        day0.times = day0.times[sel0]
        mod["pyOCEAN_hourly"] = sample_columns(hr, t, j, i, zb)
        mod["pyOCEAN_hourly_00UTC"] = sample_columns(day0, t, j, i, zb)
    Hm = H[j, i]
    jun = np.datetime64("2017-06-01", "ns")
    region = {"shelf": Hm < H_SHELF, "deep": Hm >= H_SHELF, "inner": Hm < 60.0,
              "mid": (Hm >= 60.0) & (Hm < 100.0), "outer": (Hm >= 100.0) & (Hm < H_SHELF)}
    period = {"May": t < jun, "Jun-Aug": t >= jun}
    groups = {f"{r}/{p}": rm & pm for r, rm in region.items() for p, pm in period.items()}
    groups["all"] = np.ones(t.size, bool)
    main = [k for k in mod if not k.startswith("pyOCEAN_hourly")]
    rows = band_stats(th_o, {k: mod[k][0] for k in main}, zb, groups, "theta")
    rows += band_stats(S_o, {k: mod[k][1] for k in main}, zb, groups, "S")
    # profile metrics on the same bins (model masked to where the observation has σ₀)
    m_sig = np.isfinite(sig_o)
    out = dict(t=t, lat=la, lon=lo, H=Hm, rec=rec, zb=zb, obs=(th_o, S_o, sig_o), mod=mod, groups=groups)
    met = {"obs": dict(mld=pq.mixed_layer_depth(zb, sig_o, MLD_REF, MLD_DSIG, MLD_TOP),
                       tcl=pq.thermocline_depth(zb, np.where(np.isfinite(th_o), th_o, np.nan)),
                       t05=layer_mean(th_o, zb, 0, 5), t030=layer_mean(th_o, zb, 0, 30),
                       t3060=layer_mean(th_o, zb, 30, 60), s030=layer_mean(S_o, zb, 0, 30))}
    for lab, (th_m, S_m) in mod.items():
        sg = np.where(m_sig, sigma0(th_m, S_m, zb, la, lo), np.nan)
        thm = np.where(np.isfinite(th_o), th_m, np.nan)
        met[lab] = dict(mld=pq.mixed_layer_depth(zb, sg, MLD_REF, MLD_DSIG, MLD_TOP), tcl=pq.thermocline_depth(zb, thm),
                        t05=layer_mean(thm, zb, 0, 5), t030=layer_mean(thm, zb, 0, 30),
                        t3060=layer_mean(thm, zb, 30, 60),
                        s030=layer_mean(np.where(np.isfinite(S_o), S_m, np.nan), zb, 0, 30))
    out["met"] = met
    for g, gm in groups.items():
        for key in ("mld", "tcl", "t05", "t030", "t3060", "s030"):
            ok = gm & np.isfinite(met["obs"][key])
            for lab in main:
                ok &= np.isfinite(met[lab][key])
            for lab in ["obs"] + main:
                x = met[lab][key][ok]
                d = x - met["obs"][key][ok]
                rows.append(dict(var=key, group=g, band="profile", source=lab, n=int(ok.sum()), n_prof=int(ok.sum()),
                                 median=float(np.median(x)) if x.size else np.nan,
                                 bias=float(d.mean()) if x.size else np.nan,
                                 rmse=float(np.sqrt(np.mean(d ** 2))) if x.size else np.nan))
    out["rows_meta"] = dict(n_accepted=int(a["prof_t"].size), n_used=int(t.size),
                            n_outside=int((~inbox).sum()), n_dry=int((inbox & (dist > 1.5 * dx)).sum()))
    return rows, out


def mld_test(f: dict, a: str = "pyOCEAN", b: str = "MOM6_jerlov", group: str = "shelf/Jun-Aug") -> dict:
    """The hypothesis 'pyOCEAN's warmer shelf surface comes from a shallower mixed layer than MOM6 (KPP vs ePBL)'
    on the observed profiles of a group: medians of MLD and of the 0–5, 0–30 and 30–60 m mean θ for the
    observations and both models, the across-profile correlation of Δθ₀₋₅ with ΔMLD (a − b), and of the 0–30 m mean
    Δθ with the 0–30 m mean ΔS (a − b): a column heat excess carried by salt cannot come from vertical mixing or the
    surface heat flux, it is lateral advection."""
    met, gm = f["met"], f["groups"][group]
    keys = ("mld", "tcl", "t05", "t030", "t3060", "s030")
    ok = gm.copy()                                 # profiles where MLD and 0–5 m θ exist in all three
    for s in ("obs", a, b):
        for k in ("mld", "t05"):
            ok &= np.isfinite(met[s][k])
    res = dict(group=group, n=int(ok.sum()))
    for k in keys:                                 # each metric on the subset where it exists in all three
        okk = ok.copy()
        for s in ("obs", a, b):
            okk &= np.isfinite(met[s][k])
        res[f"n_{k}"] = int(okk.sum())
        for s in ("obs", a, b):
            res[f"{s}_{k}_median"] = float(np.median(met[s][k][okk])) if okk.any() else np.nan
            res[f"{s}_{k}_mean"] = float(np.mean(met[s][k][okk])) if okk.any() else np.nan
    if ok.sum() > 5:
        dm = met[a]["mld"][ok] - met[b]["mld"][ok]
        dt = met[a]["t05"][ok] - met[b]["t05"][ok]
        res["corr_dT05_dMLD"] = float(np.corrcoef(dt, dm)[0, 1])
        res["frac_a_shallower"] = float(np.mean(dm < 0))
        dmo = met[a]["mld"][ok] - met["obs"]["mld"][ok]
        res["frac_a_shallower_than_obs"] = float(np.mean(dmo < 0))
        res["frac_b_shallower_than_obs"] = float(np.mean(met[b]["mld"][ok] - met["obs"]["mld"][ok] < 0))
        o2 = ok & np.isfinite(met[a]["t030"]) & np.isfinite(met[b]["t030"]) & np.isfinite(met[a]["s030"]) \
            & np.isfinite(met[b]["s030"])
        if o2.sum() > 5:
            res["corr_dT030_dS030"] = float(np.corrcoef(met[a]["t030"][o2] - met[b]["t030"][o2],
                                                        met[a]["s030"][o2] - met[b]["s030"][o2])[0, 1])
    return res


def _readers(py_run: Path, mom6_runs: dict[str, Path]) -> dict[str, _Reader]:
    return {"pyOCEAN": PyoceanReader(py_run), **{lab: Mom6Reader(r) for lab, r in mom6_runs.items()},
            "GLORYS12": GlorysReader()}


def _site_cells(lat: np.ndarray, lon: np.ndarray, readers: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Nearest wet cell of each point on the model grid (pyOCEAN = MOM6 grid) and on GLORYS12's."""
    from validation.nwa12_cut import case as cs

    hg, H = cs.grid(cs.CASE)
    j, i, _ = nearest_wet(hg["lat_t"].values, hg["lon_t"].values, np.isfinite(H) & (H > 0), lat, lon)
    gl = readers["GLORYS12"]
    gj, gi, _ = nearest_wet(*np.meshgrid(gl.lat, gl.lon, indexing="ij"), np.isfinite(gl.get(0)["T"][0]), lat, lon)
    return {lab: ((gj, gi) if lab == "GLORYS12" else (j, i)) for lab in readers}


def _requests(sites: list[dict], readers: dict) -> dict[str, dict]:
    """Per source, the times at which each site is compared: pyOCEAN at its 00 UTC snapshots, the others at the
    centres of their daily means, within each site's record (from day 2).  Returns {source: dict(t, site)}."""
    out = {}
    for lab, rd in readers.items():
        ts, ss = [], []
        for n, s in enumerate(sites):
            tt = rd.times[(rd.times >= T0 + np.timedelta64(1, "D")) & (rd.times >= s["t"].min())
                          & (rd.times <= s["t"].max()) & (rd.times <= np.datetime64("2017-09-01T00:00", "ns"))]
            ts.append(tt)
            ss.append(np.full(tt.size, n))
        out[lab] = dict(t=np.concatenate(ts), site=np.concatenate(ss))
    return out


def _obs_at(t: np.ndarray, x: np.ndarray, tq: np.ndarray, daily: bool, min_frac: float) -> np.ndarray:
    """Observation matched to a model time: the mean over ±30 min (snapshot) or the UTC daily mean of the day of
    ``tq`` with at least ``min_frac`` of the day's nominal samples (daily mean)."""
    out = np.full(tq.size, np.nan)
    step = np.median(np.diff(t)) if t.size > 1 else np.timedelta64(1, "h")
    win = np.timedelta64(30, "m") if not daily else np.timedelta64(12, "h")
    nominal = 2 * win / step
    for k, q in enumerate(tq):
        lo, hi = np.searchsorted(t, q - win), np.searchsorted(t, q + win, side="left" if daily else "right")
        x_ = x[lo:hi]
        if daily and np.isfinite(x_).sum() < min_frac * nominal:
            continue
        if np.isfinite(x_).any():
            out[k] = np.nanmean(x_)
    return out


def moorings_compare(py_run: Path, mom6_runs: dict[str, Path], insitu: Path = INSITU) -> tuple[list[dict], dict]:
    """Moored CTDs (NSIF ~7 m and seafloor) against the models at the instrument depth (the model's deepest value
    where the model bed is shallower): pyOCEAN 00 UTC snapshots against the observation averaged over ±30 min,
    MOM6 and GLORYS daily means against the observed UTC daily mean (≥ 80 % of the 15-min samples).  The sensors'
    in-situ T is converted to θ (gsw)."""
    import gsw

    a = load_insitu(insitu)
    readers = _readers(py_run, mom6_runs)
    sites = []
    for n, rec in enumerate(map(str, a["moored_records"])):
        t, T, S, p = (a[f"moored{n}_{v}"] for v in ("t", "T", "S", "p"))
        lat, lon = a[f"moored{n}_pos"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SA = gsw.SA_from_SP(np.where(np.isfinite(S), S, 35.0), p, lon, lat)
            th = gsw.pt0_from_t(SA, T, p)
            z = -gsw.z_from_p(p, lat)
        sites.append(dict(rec=rec, t=t, th=th, S=S, z=float(np.nanmedian(z[np.isfinite(T)])), lat=lat, lon=lon))
    cells = _site_cells(np.array([s["lat"] for s in sites]), np.array([s["lon"] for s in sites]), readers)
    req = _requests(sites, readers)
    zb = 0.5 * (np.arange(0.0, 501.0)[1:] + np.arange(0.0, 501.0)[:-1])
    rows, figs = [], {s["rec"]: dict(t=s["t"], th=s["th"], S=s["S"], z=s["z"], mod={}) for s in sites}
    for lab, rd in readers.items():
        r = req[lab]
        j, i = (c[r["site"]] for c in cells[lab])
        th_m, S_m = sample_columns(rd, r["t"], j, i, zb)
        zq = np.array([sites[k]["z"] for k in r["site"]])
        mt, ms = at_depth(th_m, zb, zq), at_depth(S_m, zb, zq)
        for n, s in enumerate(sites):
            m = r["site"] == n
            daily = lab != "pyOCEAN"
            figs[s["rec"]]["mod"][lab] = (r["t"][m], mt[m], ms[m])
            for v, mx in (("theta", mt[m]), ("S", ms[m])):
                ob = _obs_at(s["t"], s["th"] if v == "theta" else s["S"], r["t"][m], daily, 0.8)
                rows.append(dict(record=s["rec"], depth=s["z"], var=v, source=lab,
                                 sampling="daily mean" if daily else "00UTC snapshot", **skill(ob, mx)))
    return rows, figs


def emolt_compare(py_run: Path, mom6_runs: dict[str, Path], insitu: Path = INSITU) -> tuple[list[dict], dict]:
    """eMOLT bottom temperature at the sites the QC accepted as representable, against the models at the sensor
    depth (the deepest model value where the model is shallower): pyOCEAN 00 UTC snapshots against the hourly
    sample nearest 00 UTC (±30 min), MOM6/GLORYS daily means against UTC daily means (≥ 80 % of the hours).  eMOLT
    T is in-situ; at ≤ 100 m θ − T < 0.015 °C."""
    a = load_insitu(insitu)
    from validation.nwa12_cut import case as cs
    _, H = cs.grid(cs.CASE)
    readers = _readers(py_run, mom6_runs)
    sites = []
    for n, site in enumerate(map(str, a["emolt_sites"])):
        lat, lon, dep, enters, j, i = a[f"emolt{n}_meta"]
        if enters:
            sites.append(dict(rec=site, t=a[f"emolt{n}_t"], T=a[f"emolt{n}_T"], z=dep, lat=lat, lon=lon,
                              H=float(H[int(j), int(i)])))
    if not sites:
        return [], {}
    cells = _site_cells(np.array([s["lat"] for s in sites]), np.array([s["lon"] for s in sites]), readers)
    req = _requests(sites, readers)
    zb = 0.5 * (np.arange(0.0, 501.0)[1:] + np.arange(0.0, 501.0)[:-1])
    rows, figs = [], {s["rec"]: dict(t=s["t"], T=s["T"], depth=s["z"], model_H=s["H"], mod={}) for s in sites}
    for lab, rd in readers.items():
        r = req[lab]
        j, i = (c[r["site"]] for c in cells[lab])
        th_m, _ = sample_columns(rd, r["t"], j, i, zb)
        mt = at_depth(th_m, zb, np.array([sites[k]["z"] for k in r["site"]]))
        for n, s in enumerate(sites):
            m = r["site"] == n
            daily = lab != "pyOCEAN"
            figs[s["rec"]]["mod"][lab] = (r["t"][m], mt[m])
            ob = _obs_at(s["t"], s["T"], r["t"][m], daily, 0.8)
            rows.append(dict(site=s["rec"], depth=s["z"], model_H=s["H"], source=lab,
                             sampling="daily mean" if daily else "00UTC snapshot", **skill(ob, mt[m])))
    return rows, figs


ADCP_BIN = 5.0                                         # m: depth bins of the ADCP comparison


def adcp_compare(py_run: Path, mom6_runs: dict[str, Path], insitu: Path = INSITU) -> tuple[list[dict], dict]:
    """QC'd ADCP currents (east, north) against the models on 5-m depth bins: daily means (bins with ≥ 50 % of the
    day's ensembles) against MOM6/GLORYS daily means, the ensembles within ±20 min of 00 UTC against pyOCEAN
    snapshots.  Metrics on the depth mean over the bins that observation and model share on ≥ 50 % of the days:
    mean current, RMSE of the vector, complex (vector) correlation magnitude and veering (Kundu 1976)."""
    from validation.nwa12_cut import case as cs

    a = load_insitu(insitu)
    hg, _ = cs.grid(cs.CASE)
    rot = (hg["cos_t"].values, hg["sin_t"].values)
    readers = _readers(py_run, mom6_runs)
    sites = []
    for n, rec in enumerate(map(str, a["adcp_records"])):
        t, z, u, v = (a[f"adcp{n}_{x}"] for x in ("t", "z", "u", "v"))
        lat, lon = a[f"adcp{n}_pos"]
        sites.append(dict(rec=rec, t=t, z=z, u=u, v=v, lat=lat, lon=lon))
    cells = _site_cells(np.array([s["lat"] for s in sites]), np.array([s["lon"] for s in sites]), readers)
    req = _requests(sites, readers)
    edges = np.arange(0.0, 501.0, ADCP_BIN)
    zb = 0.5 * (edges[1:] + edges[:-1])
    rows, figs = [], {s["rec"]: dict(zb=zb, mod={}) for s in sites}
    for lab, rd in readers.items():
        r = req[lab]
        j, i = (c[r["site"]] for c in cells[lab])
        ue, vn = sample_columns(rd, r["t"], j, i, zb, kind="uv", rot=None if lab == "GLORYS12" else rot)
        daily = lab != "pyOCEAN"
        for n, s in enumerate(sites):
            m = np.flatnonzero(r["site"] == n)
            b = np.searchsorted(edges, s["z"], side="right") - 1
            ob = np.full((m.size, zb.size, 2), np.nan)
            ens_t = np.unique(s["t"])
            for k, q in enumerate(r["t"][m]):
                win = np.timedelta64(12, "h") if daily else np.timedelta64(20, "m")
                lo, hi = np.searchsorted(s["t"], q - win), np.searchsorted(s["t"], q + win, side="left")
                if hi <= lo:
                    continue
                n_ens = np.unique(s["t"][lo:hi]).size
                full = (2 * win) / np.median(np.diff(ens_t))
                if daily and n_ens < 0.5 * full:
                    continue
                for c, x in enumerate((s["u"], s["v"])):
                    s_ = np.bincount(b[lo:hi], weights=x[lo:hi], minlength=zb.size)[:zb.size]
                    c_ = np.bincount(b[lo:hi], minlength=zb.size)[:zb.size]
                    ob[k, :, c] = np.where(c_ >= (0.5 * n_ens if daily else 1), s_ / np.maximum(c_, 1), np.nan)
            md = np.stack([ue[m], vn[m]], axis=-1)
            common = np.isfinite(ob).all(axis=-1) & np.isfinite(md).all(axis=-1)
            keep = common.mean(axis=0) >= 0.5
            sel = common & keep[None, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                om = np.nanmean(np.where(sel[..., None], ob, np.nan), axis=1)
                mm = np.nanmean(np.where(sel[..., None], md, np.nan), axis=1)
            ok = np.isfinite(om).all(axis=1) & np.isfinite(mm).all(axis=1) & (sel.sum(axis=1) == keep.sum())
            figs[s["rec"]]["mod"][lab] = (r["t"][m], om, mm, ok, keep)
            if ok.sum() < 5:
                continue
            wo, wm = om[ok, 0] + 1j * om[ok, 1], mm[ok, 0] + 1j * mm[ok, 1]
            ao, am = wo - wo.mean(), wm - wm.mean()
            vc = np.vdot(ao, am) / np.sqrt(np.vdot(ao, ao).real * np.vdot(am, am).real)
            zk = zb[keep]
            rows.append(dict(record=s["rec"], source=lab, sampling="daily mean" if daily else "00UTC snapshot",
                             n=int(ok.sum()), z_range=f"{zk.min() - ADCP_BIN / 2:.0f}-{zk.max() + ADCP_BIN / 2:.0f}",
                             obs_mean_u=float(wo.mean().real), obs_mean_v=float(wo.mean().imag),
                             mod_mean_u=float(wm.mean().real), mod_mean_v=float(wm.mean().imag),
                             rmse=float(np.sqrt(np.mean(np.abs(wm - wo) ** 2))), vec_corr=float(np.abs(vc)),
                             veering_deg=float(np.degrees(np.angle(vc))),
                             obs_std=float(np.sqrt(np.mean(np.abs(ao) ** 2))),
                             mod_std=float(np.sqrt(np.mean(np.abs(am) ** 2)))))
    return rows, figs


def bottom_values(h: np.ndarray, x: np.ndarray, min_h: float = H_MIN) -> np.ndarray:
    """Value of the deepest non-vanished layer (h > ``min_h``) of every column of a (nz, ny, nx) field whose layers
    are surface first; NaN in columns without such a layer."""
    ok = np.isfinite(h) & (h > min_h) & np.isfinite(x)
    nz = ok.shape[0]
    kb = nz - 1 - np.argmax(ok[::-1], axis=0)
    out = np.take_along_axis(x, kb[None], axis=0)[0]
    return np.where(ok.any(axis=0), out, np.nan)


INTRUSION_BOX = dict(h_max=60.0, lat_max=40.5, lon_max=-72.0)   # southern MAB inner shelf (NJ–Delmarva)
INTRUSION_S = 35.0                                              # bottom S above this: slope / Gulf Stream water


def shelf_intrusion(py_run: Path, mom6_runs: dict[str, Path], every: int = 5) -> list[dict]:
    """Slope / Gulf Stream water on the southern MAB inner shelf: per day (every ``every`` days) the fraction of the
    cells with H < 60 m south of 40.5 °N and west of 72 °W whose bottom salinity exceeds 35, and the median bottom θ
    and S, for pyOCEAN, each MOM6 run and GLORYS12 (bilinear to the T points).  The observed shelf bottom water is
    the cold pool (θ 7–12 °C, S 32.5–33.4 in the ud_134 and eMOLT records)."""
    from validation.nwa12_cut import case as cs

    hg, H = cs.grid(cs.CASE)
    lat, lon = hg["lat_t"].values, hg["lon_t"].values
    b = INTRUSION_BOX
    m = np.isfinite(H) & (H < b["h_max"]) & (lat < b["lat_max"]) & (lon < b["lon_max"])
    rows = []
    readers = {"pyOCEAN": PyoceanReader(py_run), **{lab: Mom6Reader(r) for lab, r in mom6_runs.items()}}
    for lab, rd in readers.items():
        for k in range(0, rd.times.size, every):
            f = rd.load(k)
            sb, tb = bottom_values(f["h"], f["S"])[m], bottom_values(f["h"], f["T"])[m]
            rows.append(dict(source=lab, t=str(rd.times[k])[:13], cells=int(m.sum()),
                             frac_S35=float(np.mean(sb > INTRUSION_S)), bottom_theta=float(np.nanmedian(tb)),
                             bottom_S=float(np.nanmedian(sb))))
    gl = GlorysReader()
    for k in range(2, gl.times.size - 2, every):
        f = gl.load(k)
        bt, bs = bottom_values(f["h"], f["T"]), bottom_values(f["h"], f["S"])
        jj = np.clip(np.searchsorted(gl.lat, lat[m]), 0, gl.lat.size - 1)
        ii = np.clip(np.searchsorted(gl.lon, lon[m]), 0, gl.lon.size - 1)
        sb, tb = bs[jj, ii], bt[jj, ii]
        rows.append(dict(source="GLORYS12", t=str(gl.times[k])[:13], cells=int(np.isfinite(sb).sum()),
                         frac_S35=float(np.nanmean(sb > INTRUSION_S)), bottom_theta=float(np.nanmedian(tb)),
                         bottom_S=float(np.nanmedian(sb))))
    return rows


# ------------------------------------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_tides(rows: list[dict], path: Path) -> None:
    plt = _plt()
    srcs = sorted({r["source"] for r in rows})
    ids = sorted({r["id"] for r in rows if r["source"] != "TPXO10"})
    names = {r["id"]: r["name"] for r in rows}
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    w = 0.8 / len(srcs)
    for n, s in enumerate(srcs):
        get = {r["id"]: r for r in rows if r["source"] == s and r["constituent"] == "M2"}
        x = np.arange(len(ids)) + (n - (len(srcs) - 1) / 2) * w
        for a_, key in zip(ax, ("vector", "amp_ratio", "dphase")):
            a_.bar(x, [get[i][key] if i in get else np.nan for i in ids], w, label=s)
    ax[0].set_ylabel("M2 vector error [m]")
    ax[1].set_ylabel("A_mod / A_obs")
    ax[1].axhline(1, color="k", lw=0.5)
    ax[2].set_ylabel("G_mod − G_obs [°]")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xticks(range(len(ids)))
    ax[2].set_xticklabels([names[i] + ("*" if i in TPXO_POOR else "") for i in ids], rotation=30, ha="right")
    ax[0].legend(fontsize=8)
    ax[0].set_title("M2 against CO-OPS harcon (* TPXO10 not representative at the gauge, QC)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_eta(figs: dict, rows_qc: dict, path: Path) -> None:
    plt = _plt()
    ids = list(figs)
    fig, ax = plt.subplots(len(ids), 1, figsize=(11, 1.6 * len(ids)), sharex=True)
    for a_, sid in zip(ax, ids):
        f = figs[sid]
        first = True
        for lab, d in f["mod"].items():
            if first:
                a_.plot(f["t"], d["obs_sub"], color="k", lw=0.8, label="obs")
                a_.plot(f["t"], d["obs_sub_ib"], color="0.6", lw=0.8, label="obs − η_IB")
                first = False
            a_.plot(f["t"], d["sub"], lw=0.8, label=lab)
        a_.set_ylabel(rows_qc[sid]["name"][:12], fontsize=7)
    ax[0].legend(fontsize=7, ncol=5, loc="upper left")
    ax[0].set_title("subtidal η (Godin), demeaned [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_sst(maps: dict, path: Path, ref: str = "GLORYS12") -> None:
    plt = _plt()
    labs = [k for k in maps if not k.startswith("_")]
    g = maps["_grid"]
    fig = plt.figure(figsize=(4 * len(labs), 7.5))
    for n, lab in enumerate(labs):
        a_ = fig.add_subplot(2, len(labs), n + 1)
        pc = a_.pcolormesh(g["lon"], g["lat"], maps[lab]["mean_bias_jja"], cmap="RdBu_r", vmin=-2, vmax=2,
                           shading="auto")
        a_.set_title(f"{lab} − {ref}, SST Jun–Aug [°C]", fontsize=8)
        fig.colorbar(pc, ax=a_, shrink=0.8)
    a_ = fig.add_subplot(2, 1, 2)
    for lab in labs:
        for reg, ls in (("interior_shelf", "-"), ("interior_deep", "--")):
            t, b = maps[lab]["bias_t"][reg]
            a_.plot(t, b, ls, lw=1, label=f"{lab} {reg}")
    a_.axhline(0, color="k", lw=0.5)
    a_.set_ylabel(f"area-mean SST bias vs {ref} [°C]")
    a_.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_buoy(f: dict, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    ax[0].plot(f["tb"], f["xb"], color="k", lw=0.6, label="44025 WTMP (QC accepted)")
    ax[0].plot(f["tb"], f["xh"], color="C0", lw=0.6, label="pyOCEAN top layer")
    ax[0].legend(fontsize=7)
    ax[0].set_ylabel("hourly SST [°C]")
    ax[1].plot(f["db"].astype("datetime64[ns]") + 12 * HOUR, f["mb"], "k.-", lw=0.8, label="44025")
    for lab, d in f["est"].items():
        ax[1].plot(f["db"].astype("datetime64[ns]") + 12 * HOUR, [d.get(x, np.nan) for x in f["db"]], lw=0.8,
                   label=lab)
    ax[1].legend(fontsize=7, ncol=4)
    ax[1].set_ylabel("daily-mean SST [°C]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_oisst_qc(f: dict, path: Path) -> None:
    """OISST QC figure: pixel status, mean analysis error, mean OISST − GLORYS12 and the 44025 daily series."""
    plt = _plt()
    q, o = f["q"], f["o"]
    fig = plt.figure(figsize=(14, 8))
    code = np.select([q["pixel"] == "land", q["pixel"] == "ok", q["pixel"] == "flagged"], [np.nan, 0.0, 2.0], 1.0)
    for n, (fld, ttl, kw) in enumerate((
            (code, "pixel: 0 ok, 1 unverified, 2 flagged", dict(cmap="viridis", vmin=0, vmax=2)),
            (np.nanmean(o["err"], axis=0), "mean analysis error err [°C]", dict(cmap="magma_r", vmin=0, vmax=0.6)),
            (q["dep"], "mean OISST − GLORYS12 [°C]", dict(cmap="RdBu_r", vmin=-3, vmax=3)))):
        a_ = fig.add_subplot(2, 3, n + 1)
        pc = a_.pcolormesh(o["lon"], o["lat"], np.where(q["pixel"] == "land", np.nan, fld), shading="auto", **kw)
        a_.set_title(ttl, fontsize=8)
        fig.colorbar(pc, ax=a_, shrink=0.8)
    a_ = fig.add_subplot(2, 1, 2)
    td = f["db"].astype("datetime64[ns]") + 12 * HOUR
    a_.plot(td, f["mb"], "k.-", lw=0.8, label="44025 daily mean (QC)")
    a_.plot(o["t"], f["po"], lw=0.9, label="OISST v2.1")
    a_.plot(o["t"], f["pg"], lw=0.9, label="GLORYS12")
    a_.set_ylabel("SST at 44025 [°C]")
    a_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_profiles(rows: list[dict], path: Path, group_list=("inner/Jun-Aug", "outer/Jun-Aug", "deep/Jun-Aug")) -> None:
    """θ and S bias and RMSE against the accepted profiles per depth band (common samples)."""
    import pandas as pd
    plt = _plt()
    df = pd.DataFrame([r for r in rows if r["band"] != "profile"])
    fig, ax = plt.subplots(2, 2 * len(group_list), figsize=(4.2 * len(group_list) * 2 / 2, 8), sharey=True)
    bands = [f"{a:.0f}-{b:.0f}" for a, b in DEPTH_BANDS]
    ymid = np.array([0.5 * (a + b) for a, b in DEPTH_BANDS])
    for c, g in enumerate(group_list):
        for r_, var in enumerate(("theta", "S")):
            for k, key in enumerate(("bias", "rmse")):
                a_ = ax[r_, 2 * c + k]
                for src in df.source.unique():
                    d = df[(df["var"] == var) & (df.group == g) & (df.source == src)].set_index("band")
                    y = np.array([d[key].get(b, np.nan) for b in bands])
                    a_.plot(y, ymid, "o-", ms=3, lw=1, label=src)
                a_.axvline(0, color="k", lw=0.5)
                a_.set_title(f"{g} {var} {key}", fontsize=8)
                a_.set_yscale("symlog", linthresh=30)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_ylabel("depth [m]")
    ax[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_mld(f: dict, path: Path, group: str = "shelf/Jun-Aug") -> None:
    """Observed vs model MLD and 0–5 m θ on the profiles of a group, and the daily medians in time."""
    plt = _plt()
    met, gm = f["met"], f["groups"][group]
    srcs = [k for k in met if k != "obs" and not k.startswith("pyOCEAN_hourly")]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ok_all = gm & np.isfinite(met["obs"]["mld"])
    for s in srcs:
        ok = ok_all & np.isfinite(met[s]["mld"])
        ax[0, 0].plot(met["obs"]["mld"][ok], met[s]["mld"][ok], ".", ms=2, alpha=0.4, label=s)
        ok2 = gm & np.isfinite(met["obs"]["t05"]) & np.isfinite(met[s]["t05"])
        ax[0, 1].plot(met["obs"]["t05"][ok2], met[s]["t05"][ok2], ".", ms=2, alpha=0.4, label=s)
    for a_, lim in ((ax[0, 0], (0, 40)), (ax[0, 1], (8, 27))):
        a_.plot(lim, lim, "k", lw=0.6)
        a_.set_xlim(lim)
        a_.set_ylim(lim)
        a_.legend(fontsize=7, markerscale=4)
    ax[0, 0].set_xlabel("observed MLD [m] (σ₀ + 0.03 from 5 m)")
    ax[0, 0].set_ylabel("model MLD [m]")
    ax[0, 1].set_xlabel("observed θ 0–5 m [°C]")
    ax[0, 1].set_ylabel("model θ 0–5 m [°C]")
    day = f["t"].astype("datetime64[D]")
    for s in ["obs"] + srcs:
        for a_, key in ((ax[1, 0], "mld"), (ax[1, 1], "t05")):
            ok = gm & np.isfinite(met[s][key]) & np.isfinite(met["obs"][key])
            ud = np.unique(day[ok])
            med = [np.median(met[s][key][ok & (day == d)]) for d in ud]
            a_.plot(ud, med, "k.-" if s == "obs" else ".-", lw=0.8, ms=3, label=s)
    ax[1, 0].set_ylabel("daily median MLD [m]")
    ax[1, 1].set_ylabel("daily median θ 0–5 m [°C]")
    ax[1, 0].legend(fontsize=7)
    fig.suptitle(f"{group}: profiles with an observed MLD", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_moorings(figs: dict, path: Path) -> None:
    plt = _plt()
    ks = list(figs)
    fig, ax = plt.subplots(len(ks), 2, figsize=(14, 1.8 * len(ks) + 1), sharex=True)
    for n, k in enumerate(ks):
        f = figs[k]
        for c, (v, key) in enumerate((("th", 1), ("S", 2))):
            a_ = ax[n, c]
            a_.plot(f["t"], f[v], color="k", lw=0.4, label="obs (QC accepted)")
            for lab, m in f["mod"].items():
                a_.plot(m[0], m[key], lw=0.8, label=lab)
            a_.set_ylabel(f"{k[:8]} {f['z']:.0f} m", fontsize=7)
    ax[0, 0].legend(fontsize=6, ncol=3)
    ax[0, 0].set_title("θ [°C]", fontsize=8)
    ax[0, 1].set_title("S_P", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_emolt(figs: dict, path: Path) -> None:
    plt = _plt()
    ks = list(figs)
    fig, ax = plt.subplots(len(ks), 1, figsize=(12, 1.8 * len(ks) + 1), sharex=True, squeeze=False)
    for a_, k in zip(ax[:, 0], ks):
        f = figs[k]
        a_.plot(f["t"], f["T"], color="k", lw=0.4, label="eMOLT (QC accepted)")
        for lab, m in f["mod"].items():
            a_.plot(m[0], m[1], lw=0.8, label=lab)
        a_.set_ylabel(f"{k}\n{f['depth']:.0f} m (model {f['model_H']:.0f})", fontsize=6)
    ax[0, 0].legend(fontsize=6, ncol=3)
    ax[0, 0].set_title("bottom T [°C]", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_adcp(figs: dict, path: Path) -> None:
    plt = _plt()
    ks = list(figs)
    fig, ax = plt.subplots(len(ks), 2, figsize=(14, 1.8 * len(ks) + 1), sharex=True)
    for n, k in enumerate(ks):
        for lab, (t, om, mm, ok, keep) in figs[k]["mod"].items():
            if lab in ("MOM6", "pyOCEAN") or n == 0:
                pass
            for c in range(2):
                if lab == "MOM6_jerlov":
                    ax[n, c].plot(t[ok], om[ok, c], "k", lw=0.6, label="ADCP daily")
                ax[n, c].plot(t[ok], mm[ok, c], lw=0.7, label=lab)
        ax[n, 0].set_ylabel(k[:8] + " " + k[9:14], fontsize=6)
    ax[0, 0].legend(fontsize=6, ncol=3)
    ax[0, 0].set_title("depth-mean east velocity [m/s] (common bins)", fontsize=8)
    ax[0, 1].set_title("depth-mean north velocity [m/s]", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _csv(rows: list[dict], path: Path) -> None:
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.5g")


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    R = RUNS / "runs"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tides-py", type=Path, nargs="+", default=[R / "tides_dt600_bih", R / "tides_dt600_r21"])
    ap.add_argument("--mom6-tides", type=Path, default=R / "mom6_tides")
    ap.add_argument("--full-py", type=Path, default=R / "full_dt600_bih123")
    ap.add_argument("--mom6-full", type=Path, nargs="+", default=[R / "mom6_full", R / "mom6_full_jerlov"])
    ap.add_argument("--hourly-py", type=Path, default=R / "full_dt600_bih30h",
                    help="hourly-output run for the snapshot-vs-daily-mean effect (OISST section)")
    ap.add_argument("--oisst", type=Path, default=OISST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--insitu", type=Path, default=INSITU, help="accepted samples of obs_qc_insitu.py")
    ap.add_argument("--only", choices=SECTIONS, nargs="*", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    do = set(a.only or SECTIONS)
    have_oisst = bool(oisst_files(a.oisst))
    o = load_oisst(a.oisst) if have_oisst and ({"oisst", "buoy"} & do) else None
    summ_path = a.out / "summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    mom6_full = {("MOM6" if r.name == "mom6_full" else f"MOM6_{r.name[10:] or r.name}"): r for r in a.mom6_full}
    if "tides" in do:
        runs = {f"pyOCEAN:{r.name}": r for r in a.tides_py}
        runs["MOM6:mom6_tides"] = a.mom6_tides
        rows = tides_compare(runs)
        _csv(rows, a.out / "tides.csv")
        summary["tides"] = tides_summary(rows)
        plot_tides(rows, a.out / "tides_m2.png")
    if "eta" in do:
        runs = {"pyOCEAN": a.full_py, **mom6_full}
        rows, figs = eta_compare(runs)
        _csv(rows, a.out / "eta_full.csv")
        rows_qc, _ = qc_table()
        plot_eta(figs, rows_qc, a.out / "eta_subtidal.png")
        import pandas as pd
        df = pd.DataFrame(rows)
        summary["eta_full"] = {s: {c: float(df[df.source == s][c].median()) for c in
                                   ("total_rmse", "total_corr", "sub_rmse", "sub_corr", "sub_ib_rmse", "sub_ib_corr",
                                    "sub_ib_std_ratio", "m2_vector")} for s in df.source.unique()}
    if "sst" in do:
        rows, maps = sst_compare(a.full_py, mom6_full)
        _csv(rows, a.out / "sst_glorys.csv")
        plot_sst(maps, a.out / "sst_glorys.png")
        sb = snapshot_bias(a.full_py)
        summary["sst_snapshot_minus_dailymean"] = sb
        summary["sst_glorys"] = rows
    if "buoy" in do:
        rows, f = buoy_compare(a.full_py, mom6_full, oisst=o)
        _csv(rows, a.out / "sst_44025.csv")
        plot_buoy(f, a.out / "sst_44025.png")
        summary["sst_44025"] = rows
    if "oisst" in do and o is not None:
        from validation.nwa12_cut import case as cs

        qsum, qfig = oisst_qc_report(o)
        (a.out / "oisst_qc.json").write_text(json.dumps(qsum, indent=1, default=float))
        plot_oisst_qc(qfig, a.out / "oisst_qc.png")
        hg, _ = cs.grid(cs.CASE)
        om = oisst_on_model(o, qfig["q"]["pixel"] == "ok", hg["lat_t"].values, hg["lon_t"].values)
        gl = _glorys_sst(hg["lat_t"].values, hg["lon_t"].values)
        rows, maps = sst_compare(a.full_py, mom6_full, ref=om, extra={"GLORYS12": gl})
        _csv(rows, a.out / "sst_oisst.csv")
        plot_sst(maps, a.out / "sst_oisst.png", ref="OISST v2.1")
        summary["sst_oisst"] = rows
        summary["oisst_snapshot_effect"] = oisst_snapshot_effect(a.hourly_py, a.full_py, om)
        summary["oisst_qc"] = {k: v for k, v in qsum.items() if k != "rejected_pixels"}
    if "profiles" in do:
        rows, f = profiles_compare(a.full_py, mom6_full, hourly=a.hourly_py, insitu=a.insitu)
        _csv(rows, a.out / "insitu_profiles.csv")
        plot_profiles(rows, a.out / "insitu_profiles.png")
        summary["insitu_profiles_meta"] = f["rows_meta"]
        groups = ("shelf/Jun-Aug", "shelf/May", "deep/Jun-Aug", "inner/Jun-Aug", "mid/Jun-Aug", "outer/Jun-Aug")
        summary["mld_test"] = {g: mld_test(f, group=g) for g in groups}
        summary["mld_test_mom6_manizza"] = mld_test(f, b="MOM6")
        plot_mld(f, a.out / "insitu_mld.png")
        np.savez_compressed(a.out / "insitu_profiles_sampled.npz", t=f["t"], lat=f["lat"], lon=f["lon"], H=f["H"],
                            rec=f["rec"].astype(str), zb=f["zb"], obs_th=f["obs"][0], obs_S=f["obs"][1],
                            **{f"{k}_th": v[0].astype(np.float32) for k, v in f["mod"].items()},
                            **{f"{k}_S": v[1].astype(np.float32) for k, v in f["mod"].items()},
                            **{f"met_{s_}_{k}": v for s_, m_ in f["met"].items() for k, v in m_.items()})
        hr = [k for k in f["mod"] if k.startswith("pyOCEAN_hourly")]
        if len(hr) == 2:                                   # snapshot sampling error on the profiles (May)
            m = f["groups"]["shelf/May"]
            sb = band_stats(f["mod"]["pyOCEAN_hourly_00UTC"][0], {"hourly": f["mod"]["pyOCEAN_hourly"][0]},
                            f["zb"], {"shelf/May": m, "all/May": f["t"] < np.datetime64("2017-06-01", "ns")}, "theta")
            summary["insitu_snapshot_effect"] = sb
        summary["insitu_profiles"] = [r for r in rows if r["group"].endswith("Jun-Aug") or r["group"] == "shelf/May"]
        recs = {}                                          # profiles used per record and region (for the README)
        for r_, h_, t_ in zip(f["rec"], f["H"], f["t"]):
            k = (str(r_), "shelf" if h_ < H_SHELF else "deep", "May" if t_ < np.datetime64("2017-06-01") else "Jun-Aug")
            recs[k] = recs.get(k, 0) + 1
        summary["insitu_profiles_per_record"] = [dict(record=a_, region=b_, period=c_, n=n_)
                                                 for (a_, b_, c_), n_ in sorted(recs.items())]
    if "moorings" in do:
        rows, figs = moorings_compare(a.full_py, mom6_full, insitu=a.insitu)
        _csv(rows, a.out / "insitu_moorings.csv")
        plot_moorings(figs, a.out / "insitu_moorings.png")
        summary["insitu_moorings"] = rows
    if "emolt" in do:
        rows, figs = emolt_compare(a.full_py, mom6_full, insitu=a.insitu)
        _csv(rows, a.out / "insitu_emolt.csv")
        if figs:
            plot_emolt(figs, a.out / "insitu_emolt.png")
        summary["insitu_emolt"] = rows
    if "adcp" in do:
        rows, figs = adcp_compare(a.full_py, mom6_full, insitu=a.insitu)
        _csv(rows, a.out / "insitu_adcp.csv")
        plot_adcp(figs, a.out / "insitu_adcp.png")
        summary["insitu_adcp"] = rows
    if "intrusion" in do:
        rows = shelf_intrusion(a.full_py, mom6_full)
        _csv(rows, a.out / "insitu_intrusion.csv")
        summary["shelf_intrusion"] = rows
    summ_path.write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k in ("tides", "eta_full")}, indent=1, default=float))


if __name__ == "__main__":
    main()
