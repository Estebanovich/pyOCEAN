"""
QC of the recent raw sea-level data of the Mexican stations (IOC SLSMF relay
of the SMN-UNAM network): are the gauges working *now*?

For every station and sensor over the downloaded period (``fetch_ioc.py``):

1. Reporting: last observation, coverage, duplicated / conflicting /
   unsorted time stamps, gaps.
2. Sample checks (``diagnostics.sealevel_qc``): gross errors (|h − tide| far
   beyond any surge), spikes, stuck sensor (flat line while the tide moves),
   datum steps in the residual of the station's own harmonic analysis.
3. Clock: offset τ and gain κ per 3-day window against the station's own
   historical tide (TICON-3 record at the site, the latest one not flagged
   defective) or, failing that, TPXO10.  A whole-hour τ is a time-zone
   error, a τ that jumps or drifts is a clock problem, and κ < 1 with a lag
   relative to the gauge's own history is the classic signature of a
   clogged stilling well (float gauges) — a maintenance problem.
4. Harmonic constants of the period (P1, K2 inferred from K1, S2 with TPXO
   ratios) compared with TPXO10 and TICON-3 (``diagnose_station``).
5. Twin sensors at one site (radar + float): offset, gain, lag, steps and
   drift of their aligned difference, where the ocean cancels.

Usage:  python validation/tide_gauges/qc_ioc_mexico.py [END_DATE] [DAYS]
Outputs: out/ioc_qc_series.csv, out/ioc_qc_twins.csv, out/ioc_qc_clock.csv,
         out/ioc_qc_<code>_<sensor>.png for stations with findings.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # light job: one P-core, never the E-cores

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from compare_ticon_tpxo import ATLAS, TICON, tpxo_at  # noqa: E402
from diagnostics.sealevel_qc import (  # noqa: E402
    clock_changes,
    clock_offsets,
    compare_twins,
    detect_steps,
    flag_flatlines,
    flag_spikes,
    harmonic_analysis,
    regularize,
)
from diagnostics.tide_gauges import diagnose_station, load_ticon  # noqa: E402
from forcing.sources.tpxo import read_tpxo  # noqa: E402
from forcing.tides import CONSTITUENTS, predict  # noqa: E402

CACHE = Path.home() / "pyOCEAN_refs" / "sealevel" / "ioc"
OUT = Path(__file__).parent / "out"
CONS = list(CONSTITUENTS)
BOX = dict(lon=(-118.5, -86.0), lat=(14.0, 32.5))
GROSS = 2.5                       # m: |h − tide − median| beyond any surge on these coasts
SILENT = timedelta(days=2)        # no data for longer than this at the end date → station down
SITE_KM = 1.0                     # sensors closer than this are twins
TICON_KM = 5.0                    # a TICON record closer than this describes the same site
TIDAL_DAY = np.timedelta64(89424, "s")   # 24.84 h: long window of the step detector
_MAD = 1.4826
MIN_SKILL = 0.8                   # clock windows explaining less variance than this are not used
STEP_TAG = "<steps>"
DUPLICATE_RMS = 0.01              # m: twins closer than this are one sensor relayed twice
NOT_SEA_LEVEL = {"bat"}           # IOC sensor codes that are not sea level (battery voltage)


def km(lat1, lon1, lat2, lon2):
    return 111.2 * np.hypot(lat1 - lat2, (lon1 - lon2) * np.cos(np.deg2rad(lat1)))


def load_raw(code: str, start: date, end: date) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All cached chunks of one code → {sensor: (t, h)} in file order (raw)."""
    rows = []
    for p in sorted((CACHE / "raw").glob(f"{code}_*.json")):
        a, b = (date(int(s[:4]), int(s[4:6]), int(s[6:])) for s in p.stem.split("_")[-2:])
        if a >= start and b <= end:
            rows += json.loads(p.read_text())
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df = df[pd.to_numeric(df["slevel"], errors="coerce").notna()]
    out = {}
    for sensor, g in df.groupby("sensor", sort=False):
        if sensor in NOT_SEA_LEVEL:
            continue
        g = g.drop_duplicates(["stime", "slevel"])          # chunk boundaries overlap
        out[sensor] = (pd.to_datetime(g["stime"]).to_numpy("datetime64[ns]"),
                       g["slevel"].astype(float).to_numpy())
    return out


def native_step(t: np.ndarray) -> np.timedelta64:
    dt = np.diff(np.unique(t)) / np.timedelta64(1, "s")
    m = np.median(dt[dt > 0])
    for s in (60, 120, 180, 300, 360, 600, 900, 3600):
        if m <= 1.25 * s:
            return np.timedelta64(s, "s")
    return np.timedelta64(int(round(m)), "s")


def resolution(h: np.ndarray) -> float:
    d = np.abs(np.diff(h[np.isfinite(h)]))
    d = d[d > 1e-6]
    return float(np.round(np.quantile(d, 0.01), 4)) if d.size else np.nan


def ticon_reference(ticon: pd.DataFrame, verdicts: pd.DataFrame, lat: float, lon: float):
    """Latest TICON record within TICON_KM not flagged defective (or None)."""
    d = km(lat, lon, ticon["lat"].to_numpy(), ticon["lon"].to_numpy())
    cand = ticon[d < TICON_KM].copy()
    if verdicts is not None and len(cand):
        bad = verdicts[verdicts["verdict"].isin(["gauge_defect_confirmed", "possible_gauge_defect"])]
        keep = []
        for _, r in cand.iterrows():
            hit = bad[(np.abs(bad["lat"] - r["lat"]) < 0.01) & (np.abs(bad["lon"] - r["lon"]) < 0.01)
                      & (bad["start"] == str(r["start"].date())) & (bad["end"] == str(r["end"].date()))]
            keep.append(hit.empty)
        cand = cand[np.array(keep, bool)]
    cand = cand[np.array([np.isfinite(h).sum() >= 6 for h in cand["hc"]], bool)] if len(cand) else cand
    if not len(cand):
        return None
    return cand.sort_values("end").iloc[-1]


def analyse_series(code, sensor, t_raw, h_raw, meta, tpxo_hc, ticon_row, t_start, t_end):
    step = native_step(t_raw)
    t, h, info = regularize(t_raw, h_raw, step, tol=0.02, min_gap=np.timedelta64(1, "h"))
    n_period = int((t_end - t_start) / step)
    res = resolution(h)
    row = dict(code=code, sensor=sensor, location=meta["Location"], lat=meta["lat"], lon=meta["lon"],
               step_min=step / np.timedelta64(1, "m"), resolution_m=res,
               first=str(t[0])[:16], last=str(t[-1])[:16],
               coverage=np.isfinite(h).sum() / n_period, n_unsorted=info["n_unsorted"],
               n_duplicate=info["n_duplicate"], n_conflict=info["n_conflict"],
               n_gaps_1h=len(info["gaps"]),
               longest_gap_h=max([g[1] / np.timedelta64(1, "h") for g in info["gaps"]], default=0.0))
    findings = []
    if (t_end - t[-1]) > np.timedelta64(SILENT):
        findings.append(f"silent since {str(t[-1])[:10]}")

    # --- sample checks against the TPXO tide ------------------------------------
    pred = predict(t, tpxo_hc, CONS) if np.all(np.isfinite(tpxo_hc)) else np.zeros(t.size)
    expected = pred + np.nanmedian(h - pred)
    gross = np.abs(h - expected) > GROSS
    h1 = np.where(gross, np.nan, h)
    spikes = flag_spikes(h1, expected=pred)
    flats = flag_flatlines(h1, resolution=res if np.isfinite(res) else 0.01, expected=pred,
                           max_samples=int(np.timedelta64(12, "h") / step))
    clean = np.where(spikes | flats, np.nan, h1)
    row.update(n_gross=int(gross.sum()), n_spikes=int(spikes.sum()),
               flat_hours=float(flats.sum() * step / np.timedelta64(1, "h")),
               valid_after_qc=np.isfinite(clean).sum() / n_period)
    valid_days = np.isfinite(h).sum() * step / np.timedelta64(1, "D")
    row["spikes_per_day"] = row["n_spikes"] / valid_days if valid_days else np.nan
    if row["n_gross"]:
        findings.append(f"{row['n_gross']} gross errors")
    if row["spikes_per_day"] > 1:
        findings.append(f"{row['spikes_per_day']:.1f} spikes/day")
    if row["flat_hours"] > 1:
        findings.append(f"stuck {row['flat_hours']:.1f} h")
    if row["n_conflict"] > 10:
        findings.append(f"{row['n_conflict']} conflicting duplicates")

    # --- harmonic analysis ------------------------------------------------------
    resid = clean - expected
    span_days = float((t[np.isfinite(clean)][-1] - t[np.isfinite(clean)][0]) / np.timedelta64(1, "D")) \
        if np.isfinite(clean).any() else 0.0
    ha = None
    if row["valid_after_qc"] > 0.4 and span_days > 29 and np.all(np.isfinite(tpxo_hc)):
        inf = {"p1": ("k1", tpxo_hc[CONS.index("p1")] / tpxo_hc[CONS.index("k1")]),
               "k2": ("s2", tpxo_hc[CONS.index("k2")] / tpxo_hc[CONS.index("s2")])}
        ha = harmonic_analysis(t, clean, CONS, inference=inf)
        resid = ha["residual"]
        m2 = CONS.index("m2")
        row.update(m2_amp_cm=100 * abs(ha["hc"][m2]), m2_phase=np.rad2deg(-np.angle(ha["hc"][m2])) % 360)
        d = diagnose_station(ha["hc"], tpxo_hc, CONS)
        row.update(vs_tpxo_rss_cm=100 * d["rss"], vs_tpxo_tau_min=d["tau_minutes"], vs_tpxo_gain=d["gain"],
                   vs_tpxo_flags=" ".join(d["flags"]))
        if abs(row["vs_tpxo_gain"]) < 0.3:
            findings.append(f"NO TIDAL SIGNAL (M2 {row['m2_amp_cm']:.1f} cm; κ vs TPXO "
                            f"{row['vs_tpxo_gain']:.2f}): sensor not following the sea")
        if ticon_row is not None:
            d = diagnose_station(ha["hc"], ticon_row["hc"], CONS)
            row.update(ticon_period=f"{ticon_row['start'].year}-{ticon_row['end'].year}",
                       vs_ticon_rss_cm=100 * d["rss"], vs_ticon_tau_min=d["tau_minutes"],
                       vs_ticon_gain=d["gain"], vs_ticon_flags=" ".join(d["flags"]))
            inst = {"timezone", "units", "inverted"} & set(d["flags"])
            if inst and abs(row["vs_tpxo_gain"]) >= 0.3:
                findings.append("vs own history: " + "+".join(sorted(inst)))

    # --- datum steps --------------------------------------------------------------
    short = max(3, int(np.timedelta64(10, "m") / step))
    long = int(TIDAL_DAY / step)
    steps = detect_steps(resid, short=short, long=long, threshold=0.05)
    row["steps"] = "; ".join(f"{str(t[s['index']])[:16]} {100 * s['size']:+.0f}cm"
                             + (f" (gap {s['gap_samples'] * step / np.timedelta64(1, 'h'):.0f}h)"
                                if s["gap_samples"] else "") for s in steps)
    if steps:
        findings.append(STEP_TAG)                      # resolved against twin sensors in main()

    # --- clock within the period: reference = the station's own constants ----------
    # (a reference with other constituent ratios, e.g. TICON from another epoch,
    # makes τ and κ oscillate with the spring–neap / K1–O1 beat cycle)
    if ha is not None:
        ref_hc, ref_name = ha["hc"], "own_ha"
    elif ticon_row is not None:
        ref_hc, ref_name = np.where(np.isfinite(ticon_row["hc"]), ticon_row["hc"], 0.0), "ticon"
    else:
        ref_hc, ref_name = tpxo_hc, "tpxo"
    clock = None
    if np.isfinite(clean).sum() > 0.5 * int(np.timedelta64(3, "D") / step) and np.all(np.isfinite(ref_hc)):
        own = ref_name == "own_ha"      # same sign and constituents as the data: κ > 0, |τ| < 7.5 h
        clock = clock_offsets(t, clean, lambda tt: predict(tt, ref_hc, CONS), window=np.timedelta64(3, "D"),
                              max_lag=np.timedelta64(450 if own else 780, "m"), min_valid=0.5,
                              positive_gain=own)
        tau, gain = clock["tau_minutes"], clock["gain"]
        good = np.isfinite(tau) & (clock["skill"] >= MIN_SKILL)
        row["clock_windows"] = f"{good.sum()}/{tau.size}"
        if good.sum() >= 2:
            days = (clock["t_center"][good] - clock["t_center"][good][0]) / np.timedelta64(1, "D")
            slope = np.polyfit(days, tau[good], 1)[0] if good.sum() >= 3 else np.nan
            med = float(np.median(tau[good]))
            row.update(clock_ref=ref_name, tau_median_min=med, tau_min_min=float(tau[good].min()),
                       tau_max_min=float(tau[good].max()), tau_slope_min_per_day=slope,
                       gain_median=float(np.median(gain[good])), gain_min=float(gain[good].min()))
            if own:
                cc = clock_changes(clock["t_center"][good], tau[good])
                row.update(tau_scatter_min=float(_MAD * np.median(np.abs(tau[good] - med))),
                           jump_min=cc["jump_minutes"], jump_z=cc["jump_z"], drift_z=cc["drift_z"])
                if cc["jump"]:
                    findings.append(f"clock jump {cc['jump_minutes']:+.0f} min at {str(cc['jump_time'])[:10]} "
                                    f"(z={cc['jump_z']:.0f})")
                if cc["drift"]:
                    findings.append(f"clock drift {cc['drift_min_per_day']:+.2f} min/day (z={cc['drift_z']:.0f})")
                gc = clock_changes(clock["t_center"][good], 100 * gain[good], min_jump=10.0, min_drift=10.0)
                if gc["jump"] or gc["drift"]:
                    findings.append("gain change " + (f"{gc['jump_minutes']:+.0f} % at {str(gc['jump_time'])[:10]}"
                                                      if gc["jump"] else f"{gc['drift_min_per_day']:+.2f} %/day"))

    # --- against the gauge's own history (TICON-3 constants vs this period's) -------
    if "vs_ticon_gain" in row and np.isfinite(row["vs_ticon_gain"]) and abs(row["vs_tpxo_gain"]) >= 0.3:
        g, tau_h = row["vs_ticon_gain"], row["vs_ticon_tau_min"]
        if 0 < g < 0.8 and tau_h > 5:
            findings.append(f"damped+lagged vs own history (κ={g:.2f}, τ={tau_h:.0f} min): "
                            "stilling well / intake clogging?")
        elif not 0.8 < g < 1.25 and "units" not in row["vs_ticon_flags"]:
            findings.append(f"gain {g:.2f} vs own history")
    row["findings"] = "; ".join(findings)
    series = dict(t=t, h=h, clean=clean, expected=expected, resid=resid, spikes=spikes, flats=flats,
                  gross=gross, steps=steps, clock=clock, step=step)
    return row, series


def plot_series(row: dict, s: dict, path: Path) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t = s["t"]
    ax[0].plot(t, s["h"], lw=0.4, color="0.3", label="raw")
    ax[0].plot(t, s["expected"], lw=0.4, color="C0", alpha=0.6, label="TPXO10")
    for m, c, lab in ((s["spikes"], "C3", "spike"), (s["flats"], "C1", "stuck"), (s["gross"], "m", "gross")):
        if m.any():
            ax[0].plot(t[m], s["h"][m], ".", ms=3, color=c, label=lab)
    ax[0].legend(loc="upper right", fontsize=7, ncol=5)
    ax[0].set_ylabel("h [m]")
    ax[0].set_title(f"{row['code']} / {row['sensor']} — {row['location']}: {row['findings'] or 'no findings'}",
                    fontsize=9)
    ax[1].plot(t, s["resid"], lw=0.4, color="0.3")
    for st in s["steps"]:
        ax[1].axvline(t[st["index"]], color="C3", lw=1)
    ax[1].set_ylabel("residual [m]")
    if s["clock"] is not None:
        c = s["clock"]
        ax[2].plot(c["t_center"], c["tau_minutes"], "o-", ms=3, label="τ [min]")
        a2 = ax[2].twinx()
        a2.plot(c["t_center"], c["gain"], "s--", ms=3, color="C1", label="κ")
        a2.set_ylabel("gain κ", color="C1")
        ax[2].set_ylabel(f"τ vs {row.get('clock_ref', '')} [min]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def same(out: dict) -> bool:
    """One sensor relayed under two codes: identical gain and centimetre-level agreement."""
    return bool(out["rms_diff"] < 0.0025 or (out["rms_diff"] < DUPLICATE_RMS and abs(out["gain"] - 1) < 0.005))


def plot_twin(name_a: str, name_b: str, grid, h1, h2, out: dict, path: Path) -> None:
    """Aligned difference of two co-located sensors (the ocean cancels)."""
    lag = int(round(out["lag_minutes"] / ((grid[1] - grid[0]) / np.timedelta64(1, "m"))))
    e = out["gain"] * np.roll(h1, lag) - h2
    e -= np.nanmedian(e)
    daily = pd.Series(e, index=grid).resample("1D").median()
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(grid, 100 * e, lw=0.3, color="0.6")
    ax.plot(daily.index, 100 * daily.values, "o-", ms=3, color="C3", label="daily median")
    ax.set_ylabel(f"κ·{name_a} − {name_b} [cm]")
    ax.set_title(f"{name_a} vs {name_b}: drift {100 * 30 * out['drift_m_per_day']:+.1f} cm/month, "
                 f"lag {out['lag_minutes']:+.1f} min, κ {out['gain']:.3f}", fontsize=9)
    lo, hi = np.nanpercentile(100 * e, [0.5, 99.5])
    ax.set_ylim(lo - 2, hi + 2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    end = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    start = end - timedelta(days=days)
    t_start, t_end = np.datetime64(start, "ns"), np.datetime64(end, "ns")
    OUT.mkdir(exist_ok=True)

    meta = pd.DataFrame(json.loads((CACHE / "stationlist.json").read_text()))
    meta = meta[meta["country"] == "MEX"].drop_duplicates("Code").set_index("Code")

    tc = read_tpxo([ATLAS / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in CONS], None,
                   ATLAS / "grid_tpxo10atlas_v2.nc", lon_range=BOX["lon"], lat_range=BOX["lat"])
    ticon = load_ticon(TICON, BOX["lon"], BOX["lat"], CONS)
    verdicts = pd.concat([pd.read_csv(OUT / f"ticon_vs_tpxo_{r}.csv")
                          for r in ("pacific_mx", "gulf_of_mexico")], ignore_index=True)

    rows, series = [], {}
    for code, m in meta.iterrows():
        data = load_raw(code, start, end)
        m = m.to_dict() | {"Code": code}
        if not data:
            rows.append(dict(code=code, sensor=m.get("sensor"), location=m["Location"], lat=m["lat"],
                             lon=m["lon"], last=str(m.get("lasttime"))[:16], coverage=0.0,
                             findings=f"no data in period (last report {str(m.get('lasttime'))[:10]})"))
            continue
        tpxo_hc, node_km = tpxo_at(tc, m["lat"], m["lon"])
        tref = ticon_reference(ticon, verdicts, m["lat"], m["lon"])
        for sensor, (t_raw, h_raw) in data.items():
            if t_raw.size < 100:
                continue
            row, s = analyse_series(code, sensor, t_raw, h_raw, m, tpxo_hc, tref, t_start, t_end)
            row["tpxo_node_km"] = round(node_km, 1)
            rows.append(row)
            series[(code, sensor)] = (row, s)
            print(f"{code:6s} {sensor:4s} cov={row['coverage']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ioc_qc_series.csv", index=False, float_format="%.4g")

    clock_rows = []
    for (code, sensor), (row, s) in series.items():
        c = s["clock"]
        if c is not None:
            for tc_, tau, g, n, sk in zip(c["t_center"], c["tau_minutes"], c["gain"], c["n_valid"], c["skill"]):
                clock_rows.append(dict(code=code, sensor=sensor, t_center=str(tc_)[:16], tau_min=tau, gain=g,
                                       skill=sk, n_valid=n, ref=row.get("clock_ref")))
    pd.DataFrame(clock_rows).to_csv(OUT / "ioc_qc_clock.csv", index=False, float_format="%.4g")

    # --- twin sensors -----------------------------------------------------------
    keys = list(series)
    twin_rows, twins = [], {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ra, sa = series[a]
            rb, sb = series[b]
            if km(ra["lat"], ra["lon"], rb["lat"], rb["lon"]) > SITE_KM:
                continue
            step = max(sa["step"], sb["step"])
            ga = regularize(sa["t"][np.isfinite(sa["clean"])], sa["clean"][np.isfinite(sa["clean"])], step)
            gb = regularize(sb["t"][np.isfinite(sb["clean"])], sb["clean"][np.isfinite(sb["clean"])], step)
            t0 = max(ga[0][0], gb[0][0])
            t1 = min(ga[0][-1], gb[0][-1])
            if t1 - t0 < np.timedelta64(3, "D"):
                continue
            n = int((t1 - t0) / step) + 1
            grid = t0 + np.arange(n) * step
            h1 = pd.Series(ga[1], index=ga[0]).reindex(grid).to_numpy()
            h2 = pd.Series(gb[1], index=gb[0]).reindex(grid).to_numpy()
            both = np.isfinite(h1) & np.isfinite(h2)
            if both.sum() < int(np.timedelta64(2, "D") / step):
                continue
            short = max(3, int(np.timedelta64(10, "m") / step))
            out = compare_twins(h1, h2, step=step, short=short, long=int(TIDAL_DAY / step), threshold=0.03)
            twin_rows.append(dict(
                site=ra["location"], a=f"{a[0]}/{a[1]}", b=f"{b[0]}/{b[1]}", overlap_days=both.sum() * step
                / np.timedelta64(1, "D"), lag_min=out["lag_minutes"], gain=out["gain"],
                offset_cm=100 * out["offset"], rms_diff_cm=100 * out["rms_diff"],
                drift_cm_per_month=100 * 30 * out["drift_m_per_day"],
                steps="; ".join(f"{str(grid[s['index']])[:16]} {100 * s['size']:+.0f}cm" for s in out["steps"]),
                relation="same sensor relayed twice" if same(out) else "independent"))
            twins[(a, b)] = dict(grid=grid, both=both, steps=[grid[s["index"]] for s in out["steps"]],
                                 independent=not same(out), out=out, h1=h1, h2=h2)
            if not same(out) and (abs(out["drift_m_per_day"]) * 30 > 0.01 or out["steps"]):
                plot_twin(f"{a[0]}/{a[1]}", f"{b[0]}/{b[1]}", grid, h1, h2, out,
                          OUT / f"ioc_qc_twin_{a[0]}_{a[1]}_{b[0]}_{b[1]}.png")
            print("twin", twin_rows[-1], flush=True)
    pd.DataFrame(twin_rows).to_csv(OUT / "ioc_qc_twins.csv", index=False, float_format="%.4g")

    # --- single-sensor step candidates: confirmed / rejected by independent twins ----
    tol = np.timedelta64(3, "h")
    for key, (row, s_) in series.items():
        if STEP_TAG not in row["findings"]:
            continue
        conf = rej = unc = 0
        for st in s_["steps"]:
            ts = s_["t"][st["index"]]
            verdicts = []
            for (a, b), tw in twins.items():
                if key not in (a, b) or not tw["independent"]:
                    continue
                near = np.abs(tw["grid"] - ts) <= np.timedelta64(1, "D")
                if near.any() and tw["both"][near].mean() > 0.5:
                    verdicts.append(any(abs(x - ts) <= tol for x in tw["steps"]))
            if not verdicts:
                unc += 1
            elif any(verdicts):
                conf += 1
            else:
                rej += 1
        parts = ([f"{conf} datum step(s) CONFIRMED by twin"] if conf else []) + \
                ([f"{rej} step candidate(s) rejected by twin"] if rej else []) + \
                ([f"{unc} unconfirmed step candidate(s) (no independent twin)"] if unc else [])
        row["findings"] = row["findings"].replace(STEP_TAG, "; ".join(parts))
        df.loc[(df["code"] == key[0]) & (df["sensor"] == key[1]), "findings"] = row["findings"]
    df.to_csv(OUT / "ioc_qc_series.csv", index=False, float_format="%.4g")
    for key, (row, s_) in series.items():
        if row["findings"]:
            plot_series(row, s_, OUT / f"ioc_qc_{key[0]}_{key[1]}.png")
        print(f"{key[0]:6s} {key[1]:4s} {row['findings']}")


if __name__ == "__main__":
    main()
