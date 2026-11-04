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
   the per-step series and reported next to it.  OISST v2.1 is not downloaded (pending Esteban's authorisation).
4. **NDBC 44025 SST** (``WTMP``, hourly, the QC's accepted hours): pyOCEAN per-step top-layer T at the buoy's cell
   (hourly and daily means), MOM6 daily-mean ``tos`` at the same cell, GLORYS at the buoy.  The buoy has no
   salinity and no subsurface temperature (``44025o2017`` does not exist), so S and T(z) at 44025 are not compared.

Outputs (never in the repo): ``figs/obs_compare/`` — ``tides.csv``, ``eta_full.csv``, ``sst_glorys.csv``,
``sst_44025.csv``, ``summary.json`` and figures.

Usage:
    python validation/nwa12_cut/compare.py [--tides-py RUN …] [--full-py RUN] [--mom6-tides RUN] [--mom6-full RUN …]
"""
from __future__ import annotations

import argparse
import json
import sys
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
    """UTC daily means of a series: (day starts datetime64[D], mean), NaN where fewer than ``min_count`` finite
    samples fall in the day."""
    d = np.asarray(t, "datetime64[ns]").astype("datetime64[D]")
    x = np.asarray(x, float)
    days = np.unique(d)
    out = np.full(days.size, np.nan)
    for k, day in enumerate(days):
        v = x[d == day]
        v = v[np.isfinite(v)]
        if v.size >= min_count:
            out[k] = v.mean()
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


def model_eta(run: Path) -> tuple[np.ndarray, list[str], np.ndarray, int, float]:
    """(hourly times, station names, η (nt, n), n_avg, Δt [s]) of a run: pyOCEAN ``series.npz`` (instantaneous
    step-end values, the on-the-hour ones kept; n_avg = 1) or MOM6 ``*ocean_hourly*.nc`` SSH at the same cells, which
    is the mean of the n_avg = dt_cpld/DT step-end values of the coupling hour (``undo_step_average``)."""
    run = Path(run)
    if any(run.glob("*ocean_hourly*.nc")):
        import re
        from validation.nwa12_cut.check_mom6 import mom6_eta
        sec, names, eta = mom6_eta(run)
        cpl = float(re.search(r"dt_cpld\s*=\s*([0-9.]+)", (run / "input.nml").read_text()).group(1))
        dt = float(re.search(r"^DT\s*=\s*([0-9.]+)", (run / "MOM_parameter_doc.short").read_text(), re.M).group(1))
        n_avg = int(round(cpl / dt))
    else:
        s = np.load(run / "series.npz")
        sec, names, eta = s["t"], [str(x) for x in s["names"]], s["eta"]
        n_avg, dt = 1, float(np.median(np.diff(s["t"])))
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


def sst_compare(py_run: Path, mom6_runs: dict[str, Path]) -> tuple[list[dict], dict]:
    """Daily domain SST metrics against GLORYS12 on the common wet cells where GLORYS is defined every day, for the
    whole cut and its interior (≥ 10 cells from the open edges) split into shelf (H < 200 m) and deep water."""
    from validation.nwa12_cut import case as cs

    hg, H = cs.grid(cs.CASE)
    lat2, lon2 = hg["lat_t"].values, hg["lon_t"].values
    area = hg["area_t"].values
    tg, gl = _glorys_sst(lat2, lon2)
    src = {"pyOCEAN": _pyocean_sst(py_run)}
    src.update({lab: _mom6_sst(r) for lab, r in mom6_runs.items()})
    wet = np.isfinite(H) & np.all(np.isfinite(gl), axis=0)
    ny, nx = H.shape
    jj, ii = np.mgrid[0:ny, 0:nx]
    interior = (jj >= 10) & (jj < ny - 10) & (ii >= 10) & (ii < nx - 10)
    regions = {"all": wet, "interior_shelf": wet & interior & (H < 200), "interior_deep": wet & interior & (H >= 200)}
    rows, maps = [], {}
    t_lo = T0 + np.timedelta64(1, "D")
    for lab, (t, f) in src.items():
        keep = (t >= t_lo) & (t <= np.datetime64("2017-09-01T00:00", "ns"))
        t, f = t[keep], f[keep]
        g = _interp_time(tg, gl, t)
        d = f - g
        for reg, m in regions.items():
            w = np.where(m, area, 0.0)
            bias_t = np.nansum(np.where(m, d, 0.0) * w, axis=(1, 2)) / w.sum()
            rmse_t = np.sqrt(np.nansum(np.where(m, d, 0.0) ** 2 * w, axis=(1, 2)) / w.sum())
            for per, sel in (("May", t < np.datetime64("2017-06-01", "ns")),
                             ("Jun-Aug", t >= np.datetime64("2017-06-01", "ns")), ("all", np.ones(t.size, bool))):
                rows.append(dict(source=lab, region=reg, period=per, n_days=int(sel.sum()),
                                 bias=float(bias_t[sel].mean()), rmse=float(np.sqrt((rmse_t[sel] ** 2).mean())),
                                 cells=int(m.sum())))
            maps.setdefault(lab, {})["bias_t"] = maps.get(lab, {}).get("bias_t", {})
            maps[lab]["bias_t"][reg] = (t, bias_t)
        jja = t >= np.datetime64("2017-06-01", "ns")
        maps[lab]["mean_bias_jja"] = np.where(wet, np.nanmean(d[jja], axis=0), np.nan)
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


def buoy_compare(py_run: Path, mom6_runs: dict[str, Path], station: str = "ndbc_44025") -> tuple[list[dict], dict]:
    """44025 SST: daily means (buoy days with ≥ 20 accepted hours) of the buoy, pyOCEAN (all steps of the day), MOM6
    (daily tos at the same cell) and GLORYS (bilinear at the buoy); pyOCEAN hourly against the buoy hourly; mean
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


def plot_sst(maps: dict, path: Path) -> None:
    plt = _plt()
    labs = [k for k in maps if not k.startswith("_")]
    g = maps["_grid"]
    fig = plt.figure(figsize=(4 * len(labs), 7.5))
    for n, lab in enumerate(labs):
        a_ = fig.add_subplot(2, len(labs), n + 1)
        pc = a_.pcolormesh(g["lon"], g["lat"], maps[lab]["mean_bias_jja"], cmap="RdBu_r", vmin=-2, vmax=2,
                           shading="auto")
        a_.set_title(f"{lab} − GLORYS12, SST Jun–Aug [°C]", fontsize=8)
        fig.colorbar(pc, ax=a_, shrink=0.8)
    a_ = fig.add_subplot(2, 1, 2)
    for lab in labs:
        for reg, ls in (("interior_shelf", "-"), ("interior_deep", "--")):
            t, b = maps[lab]["bias_t"][reg]
            a_.plot(t, b, ls, lw=1, label=f"{lab} {reg}")
    a_.axhline(0, color="k", lw=0.5)
    a_.set_ylabel("area-mean SST bias vs GLORYS12 [°C]")
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
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--only", choices=("tides", "eta", "sst", "buoy"), nargs="*", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    do = set(a.only or ("tides", "eta", "sst", "buoy"))
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
        rows, f = buoy_compare(a.full_py, mom6_full)
        _csv(rows, a.out / "sst_44025.csv")
        plot_buoy(f, a.out / "sst_44025.png")
        summary["sst_44025"] = rows
    summ_path.write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k in ("tides", "eta_full")}, indent=1, default=float))


if __name__ == "__main__":
    main()
