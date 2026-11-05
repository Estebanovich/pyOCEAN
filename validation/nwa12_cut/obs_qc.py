#!/usr/bin/env python
"""
Observation QC for the NWA12 cut (regional chain, Task 11 step 0): every record used as truth is classified as
usable, partial (usable on the listed interval / with listed samples excluded) or corrupted, with the reason.

Gauges (NOAA CO-OPS, ``data/tides``: verified 6-min ``wl_<id>_<month>.json`` and hourly ``wlh_<id>_<month>.json``,
MSL datum, GMT, metres; constants ``harcon_<id>.json``), 2017-05-01 → 2017-09-01:

1. CO-OPS quality flags.  6-min ``f`` = "O|I, F, R, L" (3σ-outlier count of the 1-s samples in preliminary data,
   *inferred* in verified data; flat tolerance; rate of change; limit), hourly ``f`` = "I, L".  In these verified
   files the first 6-min flag is set exactly at the hours whose hourly height carries "I" (all 4 stations checked,
   e.g. Nantucket 63/63), so it is the inferred flag: every inferred sample (CO-OPS gap fill, not a measurement of
   the primary sensor at that time) is excluded.  F, R and L samples are kept — verified data whose 1-s test
   tripped — and judged by our own tests (L marks high waters at The Battery).
2. Gaps (``regularize``), gross errors (|h − tide − median| > ``GROSS``), spikes, stuck values, datum steps
   (``diagnostics.sealevel_qc``), the reference tide being the station's own CO-OPS constants.
3. Clock: 3-day windows of h ≈ a + κ p(t − τ) against the period's own harmonic analysis (``clock_offsets``,
   ``clock_changes``).
4. Harmonic consistency: least-squares analysis of the period (M2, S2, N2, K1, O1, Q1; K2 and P1 inferred from
   S2 and K1 with the CO-OPS ratios, the record is shorter than the 183-d Rayleigh period) compared with the CO-OPS
   ``harcon`` constants and with TPXO10-atlas-v2 at the nearest wet atlas node (``diagnose_station``: timezone,
   units, inverted, lag_damping, misfit).
5. Hourly vs 6-min: the hourly heights must equal the 6-min value on the hour (CO-OPS derives them from the same
   primary sensor; no twin sensor is in the files, so ``compare_twins`` cannot be applied — stated in the README).

Writes ``coops_qc.csv``, ``coops_qc.json`` (excluded intervals per station) and figures to ``--out``.

Usage: python validation/nwa12_cut/obs_qc.py [--tides DIR] [--tpxo DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diagnostics.sealevel_qc import (  # noqa: E402
    clock_changes,
    clock_offsets,
    detect_steps,
    flag_flatlines,
    flag_spikes,
    harmonic_analysis,
    regularize,
)
from diagnostics.tide_gauges import complex_from_amp_phase, diagnose_station  # noqa: E402
from forcing.tides import predict  # noqa: E402

RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut")
TIDES = RUNS / "data" / "tides"
TPXO = Path.home() / "pyOCEAN_refs" / "TPXO10_atlas_v2_nc"
OUT = RUNS / "figs" / "obs_qc"
GLORYS = RUNS / "data" / "glorys" / "glorys12_my_20170429_20170903_box.nc"
T0, T1 = np.datetime64("2017-05-01T00:00", "ns"), np.datetime64("2017-09-02T00:00", "ns")
STEP = np.timedelta64(6, "m")
CONS = ["m2", "s2", "n2", "k1", "o1", "q1", "k2", "p1"]
INFER = {"k2": "s2", "p1": "k1"}         # 123-d record: K2–S2 and P1–K1 need 183 d (Rayleigh R = 1)
GROSS = 1.5                              # m: |h − tide − median| beyond any summer surge here
TIDAL_DAY = np.timedelta64(89424, "s")
MIN_SKILL = 0.8
GAP_PARTIAL = np.timedelta64(6, "h")     # a gap at least this long makes the record "partial"
MAX_EXCLUDED = 0.05                      # more than this fraction excluded/missing makes it "partial"
STUCK_MIN = 4                            # identical consecutive 6-min values before a run can be "stuck"
STEP_MATCH = 30                          # samples (3 h): a step in A − B this close confirms one in A
NEIGHBOURS = 2                           # nearest gauges used to confirm a step candidate


# ------------------------------------------------------------------------------------------------ loading
def load_coops(directory: Path, sid: str, kind: str) -> tuple[dict, pd.DataFrame]:
    """Concatenate the monthly CO-OPS JSON of one product ('wl' 6-min or 'wlh' hourly): metadata, DataFrame."""
    rows, md = [], None
    for p in sorted(directory.glob(f"{kind}_{sid}_*.json")):
        j = json.loads(p.read_text())
        if "error" in j:
            raise ValueError(f"{p.name}: {j['error']}")
        md = j["metadata"]
        rows += j["data"]
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"]).to_numpy("datetime64[ns]")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    flags = np.array([list(map(int, f.split(","))) for f in df["f"]])
    for k in range(flags.shape[1]):
        df[f"f{k}"] = flags[:, k].astype(bool)
    return md, df


def harcon(directory: Path, sid: str) -> np.ndarray:
    """CO-OPS accepted constants (OTPS complex, m) for CONS."""
    hj = {c["name"].lower(): c for c in json.loads((directory / f"harcon_{sid}.json").read_text())
          ["HarmonicConstituents"]}
    return np.array([complex_from_amp_phase(hj[c]["amplitude"], hj[c]["phase_GMT"]) for c in CONS])


def tpxo_at(tpxo_dir: Path, lat: float, lon: float) -> tuple[np.ndarray, float]:
    """TPXO10 constants at the nearest atlas node with all constituents (and its distance in km)."""
    from forcing.sources.tpxo import read_tpxo, tpxo_atlas_files

    h_files, _, grid, names = tpxo_atlas_files(tpxo_dir, CONS, transports=False)
    ds = read_tpxo(h_files, None, grid, (lon - 0.3, lon + 0.3), (lat - 0.3, lat + 0.3))
    h = ds["h"].sel(constituent=CONS).values if "constituent" in ds.coords else ds["h"].values
    lo, la = np.meshgrid(ds["lon"].values, ds["lat"].values)
    ok = np.all(np.isfinite(h), axis=0)
    d = 111.2 * np.hypot((lo - lon) * np.cos(np.deg2rad(lat)), la - lat)
    d = np.where(ok, d, np.inf)
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return h[:, j, i], float(d[j, i])


# ------------------------------------------------------------------------------------------------ helpers
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _intervals(t: np.ndarray, bad: np.ndarray, min_len: int = 1) -> list[tuple[str, str]]:
    """[start, end] strings of runs of ``bad`` at least ``min_len`` samples long."""
    return [(str(t[a])[:16], str(t[b - 1])[:16]) for a, b in _runs(bad) if b - a >= min_len]


def _coherent(step: dict, resid: np.ndarray, neighbours: dict[str, np.ndarray],
              window: int = STEP_MATCH) -> str:
    """
    Reason why a single-sensor step candidate is ocean signal ('' if it is not explained).

    CO-OPS keeps one sensor per file, so the twin-sensor test is replaced by neighbours: the candidate is
    re-examined in the difference resid_A − resid_B for each neighbouring gauge B, where a coherent surge or
    frontal set-up cancels (largely) and a datum shift of A survives.  If, for some neighbour, the difference
    has no step of at least half the size within ± ``window`` samples, the jump is shared with that neighbour.
    """
    for nb, rb in neighbours.items():
        d = resid - rb
        found = detect_steps(d, short=3, long=int(TIDAL_DAY / STEP), threshold=0.05)
        same = [s for s in found if abs(s["index"] - step["index"]) <= window
                and abs(s["size"]) > 0.5 * abs(step["size"]) and np.sign(s["size"]) == np.sign(step["size"])]
        if not same:
            return f"shared with {nb}"
    return ""


def phase(h: complex) -> float:
    return float(np.rad2deg(-np.angle(h)) % 360.0)


# ------------------------------------------------------------------------------------------------ QC
def qc_station(sid: str, tides: Path, tpxo_dir: Path,
               neighbours: dict[str, np.ndarray] | None = None) -> tuple[dict, dict]:
    """QC of one CO-OPS gauge; ``neighbours`` {id: residual series on the common grid} confirm step candidates."""
    md, df = load_coops(tides, sid, "wl")
    _, dh = load_coops(tides, sid, "wlh")
    t, h, info = regularize(df["t"].to_numpy(), df["v"].to_numpy(), STEP, tol=0.002,
                            min_gap=np.timedelta64(1, "h"))
    # extend to the full period so coverage and the trailing gap count
    full = T0 + np.arange(int((T1 - T0) // STEP)) * STEP
    hv = np.full(full.size, np.nan)
    k = ((t - T0) // STEP).astype(int)
    keep = (k >= 0) & (k < full.size)
    hv[k[keep]] = h[keep]
    t, h = full, hv
    fl = {n: np.zeros(t.size, bool) for n in ("f0", "f1", "f2", "f3")}
    kk = ((df["t"].to_numpy() - T0) // STEP).astype(int)
    ok = (kk >= 0) & (kk < t.size)
    for n in fl:
        fl[n][kk[ok]] = df[n].to_numpy()[ok]

    inferred = fl["f0"].copy()
    coops_bad = inferred          # F/R/L samples are kept: verified data, the 1-s test tripped, ours decide

    hc_noaa = harcon(tides, sid)
    hc_tpxo, d_tpxo = tpxo_at(tpxo_dir, float(md["lat"]), float(md["lon"]))
    pred = predict(t, hc_noaa, CONS)
    expected = pred + np.nanmedian(h - pred)
    gross = np.abs(h - expected) > GROSS
    h1 = np.where(gross | coops_bad, np.nan, h)
    ratio = {c: (ref, hc_noaa[CONS.index(c)] / hc_noaa[CONS.index(ref)]) for c, ref in INFER.items()}
    # sample checks against the period's own tide (first pass): the 8-constituent CO-OPS prediction leaves the
    # overtides of Long Island Sound / Nantucket Sound (M4, M6) in the residual, which a 6-min spike test reads
    # as spikes.  Running median over 7 samples (42 min), spikes of ≤ 2 samples (verified data: CO-OPS already
    # removed spikes, so anything found here is a candidate).
    own = predict(t, harmonic_analysis(t, h1, CONS, inference=ratio)["hc"], CONS)
    spikes = flag_spikes(h1, expected=own, half_window=3, max_run=2)
    flats = flag_flatlines(h1, resolution=0.001, expected=own, max_samples=int(np.timedelta64(3, "h") / STEP))
    # at 1 mm resolution two equal 6-min values happen by chance near slack: a stuck run needs STUCK_MIN
    # identical consecutive values (flag_flatlines merges neighbouring equal pairs into one run)
    eq = np.zeros(h1.size, bool)
    eq[1:] = np.abs(np.diff(h1)) < 0.0005
    long_eq = np.zeros(h1.size, bool)
    for a, b in _runs(eq):
        if b - a + 1 >= STUCK_MIN:
            long_eq[a - 1:b] = True
    for a, b in _runs(flats):
        if not long_eq[a:b].any():
            flats[a:b] = False
    clean = np.where(spikes | flats, np.nan, h1)

    ha = harmonic_analysis(t, clean, CONS, inference=ratio)
    resid = ha["residual"]
    vs_noaa = diagnose_station(ha["hc"], hc_noaa, CONS)
    vs_tpxo = diagnose_station(ha["hc"], hc_tpxo, CONS)

    cand = detect_steps(resid, short=3, long=int(TIDAL_DAY / STEP), threshold=0.05)
    steps, rejected = [], []
    for st in cand:
        why = _coherent(st, resid, neighbours or {})
        (rejected if why else steps).append(st | dict(why=why))

    clock = clock_offsets(t, clean, lambda tt: predict(tt, ha["hc"], CONS), window=np.timedelta64(3, "D"),
                          max_lag=np.timedelta64(450, "m"), min_valid=0.5, positive_gain=True)
    good = np.isfinite(clock["tau_minutes"]) & (clock["skill"] >= MIN_SKILL)
    cc = clock_changes(clock["t_center"][good], clock["tau_minutes"][good])
    gc = clock_changes(clock["t_center"][good], 100 * clock["gain"][good], min_jump=10.0, min_drift=10.0)

    # hourly vs 6-min on the hour
    s6 = pd.Series(h, index=t)
    hh = dh.set_index("t")["v"]
    common = hh.index.intersection(s6.index)
    dhr = (hh.loc[common] - s6.loc[common]).to_numpy()
    hr_max = float(np.nanmax(np.abs(dhr))) if np.isfinite(dhr).any() else np.nan
    hourly_inferred = int(dh["f0"].sum())

    missing = ~np.isfinite(h)
    gaps = [(a, b) for a, b in _runs(missing) if (b - a) * STEP >= np.timedelta64(1, "h")]
    longest = max([(b - a) * STEP / np.timedelta64(1, "h") for a, b in gaps], default=0.0)
    excluded = ~np.isfinite(clean)
    coverage = float(np.isfinite(clean).mean())

    reasons, partial, corrupt = [], False, False
    if abs(vs_noaa["gain"]) < 0.3:
        corrupt = True
        reasons.append(f"no tidal signal (κ vs harcon {vs_noaa['gain']:.2f})")
    inst = {"timezone", "units", "inverted"} & set(vs_noaa["flags"])
    if inst:
        corrupt = True
        reasons.append("vs harcon: " + "+".join(sorted(inst)))
    if coverage < 0.4:
        corrupt = True
        reasons.append(f"coverage {100 * coverage:.0f} %")
    if longest * np.timedelta64(1, "h") >= GAP_PARTIAL:
        partial = True
        reasons.append("gap " + "; ".join(f"{str(t[a])[:16]}–{str(t[b - 1])[:16]} ({(b - a) * 0.1:.0f} h)"
                                          for a, b in gaps if (b - a) * STEP >= GAP_PARTIAL))
    if inferred.any():
        reasons.append(f"{int(inferred.sum())} CO-OPS-inferred samples excluded (longest run {max([(b_ - a_) for a_, b_ in _runs(inferred)]) * 0.1:.1f} h)")
    frl = fl["f1"] | fl["f2"] | fl["f3"]
    if frl.any():
        n_out = int((frl & np.isfinite(h) & ~np.isfinite(clean)).sum())
        reasons.append(f"{int(frl.sum())} F/R/L-flagged samples kept, {n_out} of them excluded (inferred or our tests)")
    if gross.sum():
        reasons.append(f"{int(gross.sum())} gross")
    if spikes.sum() or flats.sum():
        reasons.append(f"{int(spikes.sum())} spikes, {flats.sum() * 0.1:.1f} h stuck excluded")
    if steps:
        partial = True
        reasons.append("datum step " + "; ".join(f"{str(t[s['index']])[:16]} {100 * s['size']:+.1f} cm"
                                                  for s in steps))
    if rejected:
        reasons.append(f"{len(rejected)} step candidate(s) rejected (shared with a neighbour)")
    if cc["jump"] or cc["drift"]:
        partial = True
        reasons.append(f"clock {'jump %+.0f min' % cc['jump_minutes'] if cc['jump'] else 'drift %+.2f min/d' % cc['drift_min_per_day']}")
    if gc["jump"] or gc["drift"]:
        partial = True
        reasons.append("gain change")
    if np.isfinite(hr_max) and hr_max > 0.0015:
        reasons.append(f"hourly ≠ 6-min on the hour (max {100 * hr_max:.1f} cm)")
    if "misfit" in vs_noaa["flags"]:
        partial = True
        reasons.append(f"misfit vs harcon (RSS {100 * vs_noaa['rss']:.1f} cm)")
    blocks = [(b_ - a_) * STEP for a_, b_ in _runs(~np.isfinite(clean))]
    if max(blocks, default=np.timedelta64(0, "m")) >= GAP_PARTIAL or (1 - coverage) > MAX_EXCLUDED:
        partial = True
    verdict = "corrupted" if corrupt else ("partial" if partial else "usable")
    usable_from_to = (str(t[np.isfinite(clean)][0])[:16], str(t[np.isfinite(clean)][-1])[:16]) \
        if np.isfinite(clean).any() else ("", "")

    m2 = CONS.index("m2")
    row = dict(id=sid, name=md["name"], lat=float(md["lat"]), lon=float(md["lon"]), n_6min=len(df),
               coverage_raw=float(np.isfinite(h).mean()), coverage_qc=coverage, longest_gap_h=longest,
               n_inferred=int(inferred.sum()),
               n_F=int(fl["f1"].sum()), n_R=int(fl["f2"].sum()), n_L=int(fl["f3"].sum()),
               n_gross=int(gross.sum()), n_spikes=int(spikes.sum()), stuck_h=float(flats.sum() * 0.1),
               n_steps=len(steps), n_step_candidates=len(cand), clock_windows=f"{good.sum()}/{good.size}",
               tau_median_min=float(np.median(clock["tau_minutes"][good])) if good.any() else np.nan,
               tau_scatter_min=float(1.4826 * np.median(np.abs(clock["tau_minutes"][good]
                                                              - np.median(clock["tau_minutes"][good]))))
               if good.any() else np.nan,
               gain_median=float(np.median(clock["gain"][good])) if good.any() else np.nan,
               clock_jump_z=cc["jump_z"], clock_drift_z=cc["drift_z"],
               hourly_vs_6min_max_cm=100 * hr_max, hourly_inferred=hourly_inferred,
               mean_msl_cm=100 * ha["mean"],
               m2_amp_cm=100 * abs(ha["hc"][m2]), m2_phase=phase(ha["hc"][m2]),
               m2_amp_noaa_cm=100 * abs(hc_noaa[m2]), m2_phase_noaa=phase(hc_noaa[m2]),
               m2_vec_noaa_cm=100 * abs(ha["hc"][m2] - hc_noaa[m2]),
               rss_noaa_cm=100 * vs_noaa["rss"], gain_noaa=vs_noaa["gain"], tau_noaa_min=vs_noaa["tau_minutes"],
               flags_noaa=" ".join(vs_noaa["flags"]),
               tpxo_node_km=d_tpxo, m2_amp_tpxo_cm=100 * abs(hc_tpxo[m2]), m2_phase_tpxo=phase(hc_tpxo[m2]),
               rss_tpxo_cm=100 * vs_tpxo["rss"], gain_tpxo=vs_tpxo["gain"], tau_tpxo_min=vs_tpxo["tau_minutes"],
               flags_tpxo=" ".join(vs_tpxo["flags"]),
               resid_std_cm=100 * float(np.nanstd(resid)),
               verdict=verdict, reason="; ".join(reasons) or "no defect found",
               usable_from=usable_from_to[0], usable_to=usable_from_to[1],
               excluded_blocks_1h="; ".join(f"{x}–{y}" for x, y in _intervals(t, ~np.isfinite(clean), 10)))
    excl = dict(gaps=_intervals(t, missing, 10), inferred=_intervals(t, inferred),
                flagged_FRL=_intervals(t, fl["f1"] | fl["f2"] | fl["f3"]),
                spikes=_intervals(t, spikes), stuck=_intervals(t, flats), gross=_intervals(t, gross),
                steps=[(str(t[s["index"]])[:16], round(100 * s["size"], 1)) for s in steps],
                step_candidates_rejected=[(str(t[s["index"]])[:16], round(100 * s["size"], 1), s["why"])
                                          for s in rejected])
    series = dict(rejected=rejected, t=t, h=h, clean=clean, expected=expected, resid=resid, spikes=spikes, flats=flats,
                  inferred=inferred, frl=fl["f1"] | fl["f2"] | fl["f3"], steps=steps, clock=clock, good=good,
                  excluded=excluded)
    return row | dict(excluded=excl), series


# ------------------------------------------------------------------------------------------------ NDBC 44025
NDBC_POS = (40.251, -73.164)            # case.BUOY["ndbc_44025"]
# variable: (valid range, spike_min, reference departure tolerance) — the tolerance is far beyond real point-vs-grid
# differences (upwelling fronts, diurnal warm layer, 10 m vs anemometer height), so only instrument faults trip it
NDBC_CHECKS = {"WTMP": ((-2.0, 35.0), 0.5, 3.0), "ATMP": ((-20.0, 40.0), 2.0, 4.0),
               "PRES": ((950.0, 1050.0), 2.0, 5.0), "WSPD": ((0.0, 40.0), 6.0, 7.0)}


def ndbc_references(t: np.ndarray, glorys: Path, era5_dir: Path) -> dict[str, np.ndarray]:
    """Independent references at the buoy on the hourly grid: GLORYS12 top-level θ (daily mean, centred at
    12 UTC, linear in time), ERA5 t2m [°C], msl [hPa], |U10| [m/s] at the nearest 0.25° node."""
    import xarray as xr

    out = {}
    la, lo = NDBC_POS
    if glorys.exists():
        with xr.open_dataset(glorys) as g:
            sst = g["thetao"].isel(depth=0).sel(latitude=la, longitude=lo, method="nearest")
            tg = sst["time"].values + np.timedelta64(12, "h")
            out["WTMP"] = np.interp(t.astype("int64"), tg.astype("datetime64[ns]").astype("int64"), sst.values)
    files = sorted(era5_dir.glob("era5_*_instant.nc"))
    if files:
        with xr.open_mfdataset(files, combine="by_coords") as e:
            p = e.sel(latitude=la, longitude=lo, method="nearest").sel(valid_time=t, method="nearest")
            out["ATMP"] = p["t2m"].values - 273.15
            out["PRES"] = p["msl"].values / 100.0
            out["WSPD"] = np.hypot(p["u10"].values, p["v10"].values)
    return out


def qc_ndbc(path: Path, out: Path, glorys: Path, era5_dir: Path) -> list[dict]:
    """QC of the NDBC 44025 stdmet file over the comparison period; rows for the README table and a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from diagnostics.buoy_qc import flag_reference_departure, qc_buoy_series, read_ndbc_stdmet

    df = read_ndbc_stdmet(path)
    df = df[(df.index >= pd.Timestamp(T0) - pd.Timedelta("1h")) & (df.index < pd.Timestamp(T1))]
    minute = int(pd.Series(df.index.minute).mode().iloc[0])          # 44025 reports at hh:50 (end of average)
    t_start = T0 - np.timedelta64(60 - minute, "m") if minute else T0
    rows, qcs = [], {}
    refs = None
    for var, (rng, smin, tol) in NDBC_CHECKS.items():
        if var not in df:
            rows.append(dict(record=f"NDBC 44025 {var}", verdict="corrupted", reason="column absent"))
            continue
        q = qc_buoy_series(df.index.to_numpy("datetime64[ns]"), df[var].to_numpy(), rng, t_start=t_start,
                           t_end=T1, spike_min=smin)
        if refs is None:
            refs = ndbc_references(q["t"], glorys, era5_dir)
        dep = np.zeros(q["t"].size, bool)
        if var in refs:
            dep = flag_reference_departure(np.where(q["ok"], q["x"], np.nan), refs[var], min_dev=tol) & q["ok"]
        ok = q["ok"] & ~dep
        qcs[var] = (q, dep, ok)
        reasons = [f"coverage {100 * q['coverage']:.1f} %"]
        if q["gaps"]:
            reasons.append("gaps " + "; ".join(f"{str(a)[:13]} ({h:.0f} h)" for a, h in q["gaps"]))
        for name in ("range", "spike", "stuck"):
            if q[name].any():
                reasons.append(f"{int(q[name].sum())} {name}")
        if dep.any():
            reasons.append(f"{int(dep.sum())} h off the reference by > {tol:g}")
        cov = float(ok.mean())
        longest = max([h for _, h in q["gaps"]], default=0.0)
        verdict = "corrupted" if cov < 0.4 else ("partial" if longest >= 6 or 1 - cov > MAX_EXCLUDED else "usable")
        bias = float(np.nanmedian(np.where(ok, q["x"], np.nan) - refs[var])) if var in refs else np.nan
        rows.append(dict(record=f"NDBC 44025 {var}", verdict=verdict, reason="; ".join(reasons),
                         median_minus_reference=bias,
                         usable_from=str(q["t"][ok][0])[:16] if ok.any() else "",
                         usable_to=str(q["t"][ok][-1])[:16] if ok.any() else ""))
    fig, ax = plt.subplots(len(qcs), 1, figsize=(11, 2.4 * len(qcs)), sharex=True, squeeze=False)
    for a_, (var, (q, dep, ok)) in zip(ax[:, 0], qcs.items()):
        a_.plot(q["t"], q["x"], lw=0.5, color="0.3", label=f"44025 {var}")
        if var in (refs or {}):
            a_.plot(q["t"], refs[var], lw=0.6, color="C0", alpha=0.7, label="GLORYS" if var == "WTMP" else "ERA5")
        bad = ~ok & np.isfinite(q["x"])
        a_.plot(q["t"][bad], q["x"][bad], ".", ms=3, color="C3", label="excluded")
        a_.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ndbc_44025.png", dpi=110)
    plt.close(fig)
    return rows


# ------------------------------------------------------------------------------------------------ figures
def plot_station(row: dict, s: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    t = s["t"]
    ax[0].plot(t, s["h"], lw=0.3, color="0.3", label="CO-OPS 6-min (verified)")
    ax[0].plot(t, s["expected"], lw=0.3, color="C0", alpha=0.6, label="harcon prediction")
    for m, c, lab in ((s["inferred"], "C4", "inferred (excluded)"), (s["frl"], "C1", "F/R/L flag"),
                      (s["spikes"], "C3", "spike"), (s["flats"], "C2", "stuck")):
        if m.any():
            ax[0].plot(t[m], s["h"][m], ".", ms=2, color=c, label=lab)
    ax[0].legend(loc="upper right", fontsize=7, ncol=3)
    ax[0].set_ylabel("η [m, MSL]")
    ax[0].set_title(f"{row['id']} {row['name']}: {row['verdict'].upper()} — {row['reason']}", fontsize=8)
    ax[1].plot(t, s["resid"], lw=0.3, color="0.3")
    for st in s["steps"]:
        ax[1].axvline(t[st["index"]], color="C3", lw=1)
    for st in s["rejected"]:
        ax[1].axvline(t[st["index"]], color="0.6", lw=0.8, ls="--")
    ax[1].set_ylabel("residual of own HA [m]")
    c, g = s["clock"], s["good"]
    ax[2].plot(c["t_center"][g], c["tau_minutes"][g], "o-", ms=3, label="τ [min]")
    ax[2].set_ylabel("τ vs own HA [min]")
    a2 = ax[2].twinx()
    a2.plot(c["t_center"][g], c["gain"][g], "s--", ms=3, color="C1")
    a2.set_ylabel("gain κ", color="C1")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(rows: list[dict], series: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw=dict(width_ratios=[2.2, 1]))
    col = {"usable": "C2", "partial": "C1", "corrupted": "C3"}
    for k, r in enumerate(rows):
        s = series[r["id"]]
        ok = np.isfinite(s["clean"])
        for a, b in _runs(ok):
            ax[0].plot([s["t"][a], s["t"][b - 1]], [k, k], lw=6, color=col[r["verdict"]], solid_capstyle="butt")
        for a, b in _runs(~ok):
            if b - a >= 2:
                ax[0].plot([s["t"][a], s["t"][b - 1]], [k, k], lw=6, color="0.85", solid_capstyle="butt")
    ax[0].set_yticks(range(len(rows)), [f"{r['id']} {r['name']}" for r in rows], fontsize=8)
    ax[0].set_title("6-min samples entering the metrics (colour = verdict; grey = excluded)", fontsize=9)
    ax[0].invert_yaxis()
    y = np.arange(len(rows))
    ax[1].barh(y - 0.2, [r["rss_noaa_cm"] for r in rows], 0.4, label="vs CO-OPS harcon")
    ax[1].barh(y + 0.2, [r["rss_tpxo_cm"] for r in rows], 0.4, label="vs TPXO10")
    ax[1].set_yticks(y, [r["id"] for r in rows], fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("RSS of 8 constituents [cm]")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Harmonic consistency of the 2017 record", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tides", type=Path, default=TIDES)
    ap.add_argument("--tpxo", type=Path, default=TPXO)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--ndbc", type=Path, default=RUNS / "data" / "ndbc" / "44025h2017.txt.gz",
                    help="NDBC 44025 stdmet file (skipped with a message if absent)")
    ap.add_argument("--ndbc-only", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.ndbc.exists():
        rows = qc_ndbc(a.ndbc, a.out, GLORYS, RUNS / "data" / "era5")
        pd.DataFrame(rows).to_csv(a.out / "ndbc_qc.csv", index=False, float_format="%.4g")
        for r in rows:
            print(f"{r['record']:<18} {r['verdict']:<9} {r['reason']}")
    else:
        print(f"NDBC file {a.ndbc} not found: 44025 QC not run (missing data)")
    if a.ndbc_only:
        return
    ids = sorted({p.name.split("_")[1] for p in a.tides.glob("wl_*_20170501.json")})
    rows, series, excl = [], {}, {}
    first = {sid: qc_station(sid, a.tides, a.tpxo) for sid in ids}           # pass 1: residuals
    pos = {sid: (r["lat"], r["lon"]) for sid, (r, _) in first.items()}

    def km(p, q):
        return 111.2 * np.hypot((p[1] - q[1]) * np.cos(np.deg2rad(0.5 * (p[0] + q[0]))), p[0] - q[0])

    for sid in ids:
        near = sorted((s for s in ids if s != sid), key=lambda s: km(pos[sid], pos[s]))[:NEIGHBOURS]
        row, s = qc_station(sid, a.tides, a.tpxo, {n: first[n][1]["resid"] for n in near})
        row["neighbours"] = " ".join(f"{n} ({km(pos[sid], pos[n]):.0f} km)" for n in near)
        excl[sid] = row.pop("excluded")
        rows.append(row)
        series[sid] = s
        plot_station(row, s, a.out / f"coops_{sid}.png")
        print(f"{sid} {row['name']:<18} {row['verdict']:<9} {row['reason']}")
    pd.DataFrame(rows).to_csv(a.out / "coops_qc.csv", index=False, float_format="%.4g")
    (a.out / "coops_qc.json").write_text(json.dumps(dict(rows=rows, excluded=excl), indent=1, default=str))
    plot_summary(rows, series, a.out / "coops_summary.png")


if __name__ == "__main__":
    main()
