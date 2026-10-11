#!/usr/bin/env python
"""
Cross-model tidal comparison on PAC_MEX: HYCOM-IOA (expt_00.0, tiddrg=1,
drgscl=0.25, full SAL, nonlinear, baroclinic) vs pyOCEAN (barotropic, linear)
vs tide-gauge predictions (TICON-3) and raw observations (IOC/SMN), over the
hourly HYCOM archives 2026-09-26 13Z … 2026-10-03 00Z.

HYCOM srfhgt contains the non-tidal SSH too, so every series is reduced to
its tidal band by removing a mean and a linear trend over the window.  At each
point the model is fitted against a reference r(t) as
    h(t) ≈ a + b t + κ r(t − τ)                       (``clock_offsets``)
giving the gain κ, the lag τ and the residual rms; the reference is the TICON-3
prediction (gauges) or the raw IOC record (stations with data in the window).

Also checks whether each forecast cycle re-ramps the tide (blkdat tid_t0):
the two HYCOM cycles overlap 2026-09-28 13Z … 10-01 00Z.

Usage:  python validation/hycom_ioa/compare_hycom_pyocean.py [pyOCEAN tag ...]
Output: validation/hycom_ioa/out/*.csv, *.png, summary.txt
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "tide_gauges"))

from compare_ticon_tpxo import ATLAS, TICON, tpxo_at  # noqa: E402
from diagnostics.sealevel_qc import clock_offsets, regularize  # noqa: E402
from diagnostics.tide_gauges import load_ticon  # noqa: E402
from forcing.sources.tpxo import read_tpxo  # noqa: E402
from forcing.tides import CONSTITUENTS, predict  # noqa: E402

REFS = Path.home() / "pyOCEAN_refs"
CYCLES = ["0260269012", "0260271012"]
RUNS = ROOT / "validation" / "pacmex_tides" / "out"
OUT = Path(__file__).parent / "out"
CONS = list(CONSTITUENTS)
HYCOM_DAY0 = np.datetime64("1900-12-31T00:00", "s")       # HYCOM yrflag=3: day 1 = 1901-01-01
HOUR = np.timedelta64(1, "h")


def hycom_series() -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Hourly HYCOM η avoiding the first day of each forecast cycle (start-up shock of
    30–50 cm hour-to-hour jumps): cycle 269 until its end (≥ 48 h after its start),
    then cycle 271 (≥ 60 h after its start).
    """
    cyc = {}
    for c in CYCLES:
        d = np.load(REFS / f"pacmex_ssh_{c}.npz")
        t = HYCOM_DAY0 + np.round(d["model_day"] * 24).astype(np.int64) * HOUR
        cyc[c] = dict(t=t, ssh=d["ssh"], lon=d["lon"], lat=d["lat"], i0=int(d["i0"]), j0=int(d["j0"]))
    a, b = cyc[CYCLES[0]], cyc[CYCLES[1]]
    late = b["t"] > a["t"][-1]
    return np.concatenate([a["t"], b["t"][late]]), np.concatenate([a["ssh"], b["ssh"][late]]), cyc


def highpass(x: np.ndarray) -> np.ndarray:
    """Remove the subtidal signal: x − 25-h centred running mean (hourly series, axis 0); edges NaN."""
    return x - pd.DataFrame(x.reshape(x.shape[0], -1)).rolling(25, center=True, min_periods=18).mean() \
        .to_numpy().reshape(x.shape)


def fit(h: np.ndarray, t: np.ndarray, ref) -> dict:
    """κ, τ and residual rms of HP[h](t) ≈ κ HP[r](t − τ) over the window (HP = ``highpass``)."""
    h = highpass(h)
    ref0 = ref

    def ref(tt):
        full = ref0(np.arange(tt[0] - 12 * HOUR, tt[-1] + 13 * HOUR, HOUR))   # pad for the running mean
        return highpass(full)[12:12 + tt.size]
    ok = np.isfinite(h)
    if ok.sum() < 48:
        return dict(gain=np.nan, lag_min=np.nan, rms_cm=np.nan, skill=np.nan)
    trend = np.zeros_like(h)
    span = t[-1] - t[0] + HOUR
    c = clock_offsets(t, h - trend, ref, window=span, max_lag=np.timedelta64(3, "h"), min_valid=0.5,
                      positive_gain=True)
    k, tau = c["gain"][0], c["tau_minutes"][0]
    r = ref(t - np.timedelta64(int(round(tau * 60)), "s"))
    e = (h - trend) - k * r
    e = e - np.nanmean(e)
    return dict(gain=k, lag_min=tau, rms_cm=100 * np.sqrt(np.nanmean(e[ok] ** 2)), skill=c["skill"][0])


def ioc_hourly(code: str, sensor: str, t: np.ndarray) -> np.ndarray | None:
    sys.path.insert(0, str(ROOT / "validation" / "tide_gauges"))
    from qc_ioc_mexico import load_raw  # noqa: E402
    data = load_raw(code, date(2026, 9, 10), date(2026, 10, 10)).get(sensor)
    if data is None:
        return None
    tt, hh = regularize(*data, step=np.timedelta64(60, "s"))[:2]
    s = pd.Series(hh, index=tt).rolling(61, center=True, min_periods=30).median()   # hourly, spike-robust
    return s.reindex(t).to_numpy()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tags = sys.argv[1:] or ["sep_jsl0.5_cb", "sep_cd0.0025"]
    t, hy, cyc = hycom_series()
    runs = {k: np.load(RUNS / f"run_{k}.npz") for k in tags}
    r0 = runs[tags[0]]
    meta = json.loads(str(r0["meta"]))
    assert (meta["i0"], meta["j0"]) == (cyc[CYCLES[0]]["i0"], cyc[CYCLES[0]]["j0"]), "grids not aligned"
    py = {}
    for k, r in runs.items():
        st = r["snap_t"].astype("datetime64[s]")
        idx = np.searchsorted(st, t)
        ok = (idx < st.size) & (st[np.minimum(idx, st.size - 1)] == t)
        py[k] = np.where(ok[:, None, None], r["snap_eta"][np.minimum(idx, st.size - 1)], np.nan)
    # common window: hours covered by the pyOCEAN snapshots (also skips the HYCOM ramp after tid_t0)
    cov = np.isfinite(py[tags[0]]).any(axis=(1, 2))
    t, hy = t[cov], hy[cov]
    py = {k: v[cov] for k, v in py.items()}
    print(f"common window {t[0]} … {t[-1]} ({t.size} h)")

    lines = []
    # ---- 1. re-ramp check: cycle 271 vs cycle 269 over their overlap ------------
    a, b = cyc[CYCLES[0]], cyc[CYCLES[1]]
    common = np.intersect1d(a["t"], b["t"])
    ia, ib = np.searchsorted(a["t"], common), np.searchsorted(b["t"], common)
    wet = np.isfinite(a["ssh"][0])
    std_a = np.nanstd(a["ssh"][ia][:, wet], axis=0)
    rows = []
    for n, (j, k) in enumerate(zip(ia, ib)):
        d = b["ssh"][k][wet] - a["ssh"][j][wet]
        rows.append(dict(hours_since_cycle_start=n, rms_diff_cm=100 * np.sqrt(np.mean((d - d.mean()) ** 2))))
    rr = pd.DataFrame(rows)
    rr.to_csv(OUT / "cycle_overlap.csv", index=False, float_format="%.3f")
    lines.append(f"cycle 271 − cycle 269 (overlap {common.size} h): rms diff first hour "
                 f"{rr.rms_diff_cm.iloc[0]:.1f} cm, hours 1–6 {rr.rms_diff_cm.iloc[1:7].mean():.1f} cm, "
                 f"last 24 h {rr.rms_diff_cm.iloc[-24:].mean():.1f} cm (tidal std {100 * np.mean(std_a):.0f} cm)")

    # ---- 2. tide gauges (TICON-3 prediction as reference) ----------------------
    box_lon, box_lat = (-118.5, -91.5), (11.5, 32.5)
    ticon = load_ticon(TICON, box_lon, box_lat, CONS)
    tc = read_tpxo([ATLAS / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in CONS], None,
                   ATLAS / "grid_tpxo10atlas_v2.nc", lon_range=box_lon, lat_range=box_lat)
    rows = []
    seen = set()
    for key, (j, i) in zip(r0["gauge_keys"], r0["gauge_ij"]):
        lat, lon, year = key.split("_")
        lat, lon = float(lat), float(lon)
        if (round(lat, 2), round(lon, 2)) in seen:
            continue
        seen.add((round(lat, 2), round(lon, 2)))
        rec = ticon[(np.abs(ticon.lat - lat) < 1e-3) & (np.abs(ticon.lon - lon) < 1e-3)]
        if rec.empty:
            continue
        hc = np.where(np.isfinite(rec.iloc[-1]["hc"]), rec.iloc[-1]["hc"], 0.0)
        ref = lambda tt, hc=hc: predict(tt, hc, CONS)          # noqa: E731
        h_tp, _ = tpxo_at(tc, lat, lon)
        region = "Gulf of California" if (lat > 23 and lon > -114.8 and not (lat > 30 and lon < -115.5)) else "Pacific"
        row = dict(lat=lat, lon=lon, region=region, tide_range_cm=100 * np.ptp(ref(t)))
        for name, series in [("hycom", hy[:, j, i])] + [(k, py[k][:, j, i]) for k in tags] + \
                [("tpxo", predict(t, h_tp, CONS))]:
            f = fit(series.astype(float), t, ref)
            row |= {f"{name}_{q}": v for q, v in f.items()}
        rows.append(row)
    g = pd.DataFrame(rows).sort_values("lat")
    g.to_csv(OUT / "gauges_vs_ticon.csv", index=False, float_format="%.3f")
    for reg, d in g.groupby("region"):
        parts = [f"{n}: rms {d[f'{n}_rms_cm'].median():.1f} cm, κ {d[f'{n}_gain'].median():.2f}"
                 for n in ["hycom", *tags, "tpxo"]]
        lines.append(f"TICON gauges, {reg} ({len(d)}): " + " | ".join(parts))

    # ---- 3. raw observations (IOC) in the window -------------------------------
    st = pd.read_csv(ROOT / "validation" / "tide_gauges" / "out" / "ioc_qc_series.csv")
    good = st[(st.coverage > 0.5) & st.findings.isna() & st.lon.between(*box_lon) & st.lat.between(*box_lat)]
    good = good.drop_duplicates(["lat", "lon"])
    lon_t, lat_t = r0["lon_t"], r0["lat_t"]
    wet_t = np.isfinite(py[tags[0]][-1])
    rows = []
    for _, s in good.iterrows():
        obs = ioc_hourly(s.code, s.sensor, t)
        if obs is None or np.isfinite(obs).sum() < 48:
            continue
        d = np.hypot((lon_t - s.lon) * np.cos(np.deg2rad(s.lat)), lat_t - s.lat)
        d = np.where(wet_t, d, np.inf)
        j, i = np.unravel_index(np.argmin(d), d.shape)
        # the observation (with gaps) is the fitted series and the gap-free model the reference:
        # obs ≈ κ' model(t − τ'), reported as the model's gain 1/κ' and lag −τ'
        x = (t - t[0]) / HOUR
        hp = highpass(obs)
        row = dict(code=s.code, sensor=s.sensor, location=s.location, lat=s.lat, lon=s.lon,
                   obs_range_cm=100 * (np.nanmax(hp) - np.nanmin(hp)), obs_valid_h=int(np.isfinite(obs).sum()))
        for name, series in [("hycom", hy[:, j, i])] + [(k, py[k][:, j, i]) for k in tags]:
            m = series.astype(float)
            ref = lambda tt, m=m: np.interp((tt - t[0]) / HOUR, x, m)        # noqa: E731
            f = fit(obs, t, ref)
            row |= {f"{name}_gain": 1 / f["gain"], f"{name}_lag_min": -f["lag_min"],
                    f"{name}_rms_cm": f["rms_cm"], f"{name}_skill": f["skill"]}
        rows.append(row)
    o = pd.DataFrame(rows).sort_values("lat")
    o.to_csv(OUT / "ioc_obs.csv", index=False, float_format="%.3f")
    gom = (o.lon > -97.5) & (o.lat > 17.5) if len(o) else []          # Bay of Campeche piece of the subdomain
    for label, d in (("Pacific + Gulf of California", o[~gom]), ("Gulf of Mexico (near the E Flather edge)", o[gom])):
        if len(d):
            parts = [f"{n}: rms {d[f'{n}_rms_cm'].median():.1f} cm, κ {d[f'{n}_gain'].median():.2f}"
                     for n in ["hycom", *tags]]
            lines.append(f"IOC observations, {label} ({len(d)}: {', '.join(d.code)}): " + " | ".join(parts))

    # ---- 4. maps: tidal range over the window ---------------------------------
    def rng(x):
        return np.nanmax(x, axis=0) - np.nanmin(x, axis=0)
    both = np.all(np.isfinite(py[tags[0]]), axis=0)
    R_hy = np.where(both, rng(highpass(hy.astype(float))), np.nan)          # tidal band only
    R_py = rng(highpass(py[tags[0]].astype(float)))
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a_, f, ttl, kw in ((ax[0], 100 * R_hy, "HYCOM-IOA range, tidal band [cm]", dict(vmin=0, vmax=500)),
                           (ax[1], 100 * R_py, f"pyOCEAN ({tags[0]}) range [cm]", dict(vmin=0, vmax=500)),
                           (ax[2], 100 * (R_py - R_hy), "pyOCEAN − HYCOM [cm]", dict(vmin=-40, vmax=40, cmap="RdBu_r"))):
        c = a_.pcolormesh(lon_t[::2, ::2], lat_t[::2, ::2], f[::2, ::2], shading="auto", **kw)
        fig.colorbar(c, ax=a_, shrink=0.8)
        a_.set_title(ttl, fontsize=9)
        a_.set_aspect("equal")
        a_.plot(g.lon, g.lat, "k.", ms=3)
    fig.tight_layout()
    fig.savefig(OUT / "range_maps.png", dpi=100)
    plt.close(fig)

    # ---- 5. time series at Gulf of California gauges --------------------------
    gg = g[g.region == "Gulf of California"]
    fig, ax = plt.subplots(len(gg), 1, figsize=(11, 2.0 * len(gg)), sharex=True)
    for a_, (_, r) in zip(np.atleast_1d(ax), gg.iterrows()):
        key = [k for k in r0["gauge_keys"] if k.startswith(f"{r.lat:.3f}_{r.lon:.3f}")][0]
        j, i = r0["gauge_ij"][list(r0["gauge_keys"]).index(key)]
        rec = ticon[(np.abs(ticon.lat - r.lat) < 1e-3) & (np.abs(ticon.lon - r.lon) < 1e-3)].iloc[-1]
        p = predict(t, np.where(np.isfinite(rec["hc"]), rec["hc"], 0.0), CONS)
        a_.plot(t, 100 * p, "k", lw=1.5, label="TICON prediction")
        a_.plot(t, 100 * (hy[:, j, i] - np.nanmean(hy[:, j, i])), "C3", lw=1, label="HYCOM-IOA")
        a_.plot(t, 100 * (py[tags[0]][:, j, i] - np.nanmean(py[tags[0]][:, j, i])), "C0", lw=1, label="pyOCEAN")
        a_.set_ylabel("cm")
        a_.set_title(f"{r.lat:.2f}N {r.lon:.2f}E — rms vs TICON: HYCOM {r.hycom_rms_cm:.0f}, "
                     f"pyOCEAN {r[f'{tags[0]}_rms_cm']:.0f}, TPXO {r.tpxo_rms_cm:.0f} cm", fontsize=8)
    np.atleast_1d(ax)[0].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "gulf_gauges_timeseries.png", dpi=100)
    plt.close(fig)

    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
