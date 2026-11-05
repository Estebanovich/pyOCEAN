#!/usr/bin/env python
"""
Quality control of the in-situ observations of the NWA12 cut (regional chain Task 11, second observation set;
Esteban's rule R26): OOI Coastal Pioneer moorings (CTDs, wire-following profilers, ADCPs, point velocity meters,
seafloor pressure), IOOS glider DAC deployments (OOI Pioneer delayed-mode Slocum, Rutgers, U. Delaware, UMass,
WHOI Spray) and eMOLT bottom temperature, 2017-05-01 → 2017-09-01 (``data/OBS_INVENTORY.md``).

Every record is classified **usable**, **partial** (usable with listed exclusions) or **corrupted**, with the
reason, separately for T and S where both exist; only accepted samples are written to the comparison cache.  The
checks are those of ``diagnostics.profile_qc`` (Argo/QARTOD tests) and ``diagnostics.buoy_qc`` (time series):

* provider flags where present (OOI ``*_qc_agg`` QARTOD aggregate: 1, 2 kept, 3 suspect and 4 fail rejected; glider
  DAC ``*_qc`` 0/1/2/5 kept, 3/4/8/9 rejected; the OOI delayed glider files, the real-time Slocum files and the
  ADCP files carry no usable flags (absent or all NaN), so our tests alone decide there);
* gross range (regional: θ −1.5…31 °C, S 26…37.2, p −1…1100 dbar), out-of-water samples (C < 2 S m⁻¹), Argo spike
  test, stuck profiles, density inversions > 0.03 kg m⁻³ on 1-m bins (salinity only: in an unpumped Slocum CT cell
  an inversion is the thermal-lag salinity error, the temperature is kept), position/time (box, period, speed ≤
  3 m s⁻¹, duplicates, provider position flags);
* moored instruments: off-station pressure (deployment, recovery, wrong units), spikes and stuck values on the
  instrument's own grid, gaps;
* the unflagged gliders and the moored CTDs are cross-checked against independent accepted profiles (OOI profiler
  moorings and other gliders within 5 km and 2 h): median bias and Theil–Sen drift per deployment;
* eMOLT: sites by position, spikes, stuck values, handling transients (first/last 3 h of each record), and
  representativeness on the model grid (inside the cut, wet model cell within 1.5 cells, model depth within a factor
  2 of the sensor depth).

Data are opened only with xarray (run with ``python -I``); macOS ``._*`` files are skipped.

Outputs (never in the repo): ``figs/obs_qc/insitu_qc.csv`` (one row per record), ``insitu_qc.json`` (rows,
cross-check pairs, excluded intervals), ``insitu_accepted.npz`` (accepted samples for ``compare.py``),
``insitu_*.png``.

Usage:  python -I validation/nwa12_cut/obs_qc_insitu.py [--data DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diagnostics import profile_qc as pq  # noqa: E402

RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut")
DATA = RUNS / "data"
OUT = RUNS / "figs" / "obs_qc"
T0 = np.datetime64("2017-05-01T00:00", "ns")
T1 = np.datetime64("2017-09-01T00:00", "ns")             # inclusive (the requests end at 09-01 00:00)
DATA_BOX = (-76.5, -67.5, 35.5, 42.8)                     # lon0, lon1, lat0, lat1 of the requests
EDGES = np.r_[np.arange(0.0, 250.0, 1.0), np.arange(250.0, 1010.0, 10.0)]   # depth bins [m]
ZB = 0.5 * (EDGES[1:] + EDGES[:-1])
RANGE_T, RANGE_S, RANGE_P = (-1.5, 31.0), (26.0, 37.2), (-1.0, 1100.0)
SPIKE = {"T": (6.0, 2.0), "S": (0.9, 0.3)}                # Argo test 9 thresholds above / below 500 dbar
C_MIN = 2.0                                               # S m⁻¹
INVERSION = 0.03                                          # kg m⁻³ (Argo test 14)
MIN_BINS = 5                                              # accepted 1-m bins for a profile to count
#: cross-check: a deployment is biased when the median difference to independent profiles exceeds these with at
#: least CROSS_MIN pairs; drift when the Theil–Sen slope exceeds DRIFT_S per 30 days
BIAS_T, BIAS_S, DRIFT_S, CROSS_MIN = 0.2, 0.05, 0.05, 10
PAIR_KM, PAIR_DT = 5.0, np.timedelta64(2, "h")


def _files(directory: Path, pattern: str = "*.nc") -> list[Path]:
    return sorted(p for p in Path(directory).glob(pattern) if not p.name.startswith("._"))


def _record(path: Path) -> str:
    return path.stem.replace("ooi-", "")


# ---------------------------------------------------------------------------------------------------- loading
def load_glider(path: Path) -> dict:
    """Observation-level arrays of a glider DAC .ncCF file: t, p, T, S, C (S m⁻¹), pid; per profile t, lat, lon;
    provider flags (dict of arrays, NaN where absent) and the flag scheme ('dac')."""
    import xarray as xr

    with xr.open_dataset(path) as d:
        rs = d["rowSize"].values.astype(np.int64)
        pid = np.repeat(np.arange(rs.size), rs)
        g = lambda k: d[k].values.astype(float) if k in d else np.full(pid.size, np.nan)   # noqa: E731
        t = d["precise_time"].values if "precise_time" in d else d["time"].values[pid]
        t = np.where(np.isnat(t), d["time"].values[pid], t).astype("datetime64[ns]")
        o = dict(t=t, p=g("pressure"), T=g("temperature"), S=g("salinity"), C=g("conductivity"), pid=pid,
                 flags={"T": g("temperature_qc"), "S": g("salinity_qc"), "p": g("pressure_qc")},
                 qartod={"T": g("qartod_temperature_primary_flag"), "S": g("qartod_salinity_primary_flag")},
                 pos_flag=np.fmax.reduceat(np.fmax(np.fmax(g("profile_lat_qc"), g("profile_lon_qc")),
                                                   g("profile_time_qc")), np.r_[0, np.cumsum(rs)[:-1]])
                 if rs.size else np.zeros(0))
        o["prof"] = dict(t=d["time"].values.astype("datetime64[ns]"), lat=d["latitude"].values.astype(float),
                         lon=d["longitude"].values.astype(float))
    o["pos_flag"] = np.where(np.isin(o["pos_flag"], (3, 4)), 1, 0).astype(bool)
    return o


def _ooi(d, base: str) -> tuple[np.ndarray, np.ndarray]:
    """(values, qc_agg) of an OOI variable, with or without the profiler suffix."""
    for k in (base + "_profiler_depth_enabled", base):
        if k in d:
            q = d[k + "_qc_agg"].values.astype(float) if k + "_qc_agg" in d else np.full(d.sizes["row"], np.nan)
            return d[k].values.astype(float), q
    n = d.sizes["row"]
    return np.full(n, np.nan), np.full(n, np.nan)


def load_wfp(path: Path) -> dict:
    """OOI wire-following profiler CTD as profiles: legs from ``profile_qc.split_legs``."""
    import xarray as xr

    with xr.open_dataset(path) as d:
        t = d["time"].values.astype("datetime64[ns]")
        T, qT = _ooi(d, "sea_water_temperature")
        S, qS = _ooi(d, "sea_water_practical_salinity")
        p, qp = _ooi(d, "sea_water_pressure")
        lat, lon = float(d["latitude"].values[0]), float(d["longitude"].values[0])
    order = np.argsort(t, kind="stable")
    t, T, S, p, qT, qS, qp = (a[order] for a in (t, T, S, p, qT, qS, qp))
    leg = pq.split_legs(t, p, gap=np.timedelta64(20, "m"), min_span=10.0)
    keep = leg >= 0
    t, T, S, p, qT, qS, qp, leg = (a[keep] for a in (t, T, S, p, qT, qS, qp, leg))
    # rows of one leg in pressure order (the 1-min bins hold several depths in a fixed order)
    sign = np.zeros(leg.max() + 1 if leg.size else 0)
    if leg.size:
        first = np.r_[0, np.flatnonzero(np.diff(leg)) + 1]
        last = np.r_[first[1:] - 1, leg.size - 1]
        sign = np.sign(p[last] - p[first])
    order = np.lexsort((p * sign[leg], leg))
    t, T, S, p, qT, qS, qp, leg = (a[order] for a in (t, T, S, p, qT, qS, qp, leg))
    nl = int(leg.max()) + 1 if leg.size else 0
    _, first, count = np.unique(leg, return_index=True, return_counts=True)
    tl = t[first + count // 2]                                   # mid-leg time
    return dict(t=t, p=p, T=T, S=S, C=np.full(t.size, np.nan), pid=leg, flags={"T": qT, "S": qS, "p": qp},
                qartod=None, pos_flag=np.zeros(nl, bool),
                prof=dict(t=tl, lat=np.full(nl, lat), lon=np.full(nl, lon)))


# ---------------------------------------------------------------------------------------------------- profiles
def _teos(Tb: np.ndarray, Sb: np.ndarray, pb: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """(θ [°C], σ₀ [kg m⁻³]) of in-situ binned T, S_P, p; θ uses S = 35 where S is missing (θ − T changes by
    < 0.002 °C per unit of S in the upper 1000 m)."""
    import gsw

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sx = np.where(np.isfinite(Sb), Sb, 35.0)
        SA = gsw.SA_from_SP(Sx, pb, lon[:, None], lat[:, None])
        th = gsw.pt0_from_t(SA, Tb, pb)
        sig = gsw.sigma0(SA, gsw.CT_from_t(SA, Tb, pb))
    return th, np.where(np.isfinite(Sb), sig, np.nan)


def qc_profiles(o: dict, record: str, platform: str, scheme: str | None) -> tuple[dict, dict]:
    """QC of one profiling record (glider deployment or profiler mooring).

    Returns (row: counts, accepted fractions and verdicts; accepted: binned θ, S, σ₀ (n_prof, n_bins) with
    rejected bins NaN, per-profile t/lat/lon and the profile acceptance mask)."""
    import gsw

    T, S, p, pid = o["T"], o["S"], o["p"], o["pid"]
    prof = o["prof"]
    npr = prof["t"].size
    lat_o = prof["lat"][np.clip(pid, 0, max(npr - 1, 0))]
    z = -gsw.z_from_p(p, lat_o)
    raw_T, raw_S = np.isfinite(T), np.isfinite(S)
    cnt = {}

    def drop(name: str, mT: np.ndarray, mS: np.ndarray) -> None:
        cnt[name] = (int((mT & okT).sum()), int((mS & okS).sum()))

    okT, okS = raw_T.copy(), raw_S.copy()
    if scheme is not None:
        fT = ~(pq.provider_ok(o["flags"]["T"], scheme) & pq.provider_ok(o["flags"]["p"], scheme))
        fS = ~(pq.provider_ok(o["flags"]["S"], scheme) & pq.provider_ok(o["flags"]["p"], scheme))
        if o.get("qartod"):
            fT |= ~pq.provider_ok(o["qartod"]["T"], "qartod")
            fS |= ~pq.provider_ok(o["qartod"]["S"], "qartod")
        drop("provider", fT, fS)
        okT &= ~fT
        okS &= ~fS
    bp = pq.gross_range(p, *RANGE_P) | ~np.isfinite(p)
    rT, rS = pq.gross_range(T, *RANGE_T) | bp, pq.gross_range(S, *RANGE_S) | bp
    drop("range", rT, rS)
    okT &= ~rT
    okS &= ~rS
    ow = pq.out_of_water(o["C"], C_MIN)
    drop("out_of_water", ow, ow)
    okT &= ~ow
    okS &= ~ow
    sT = pq.vertical_spike(np.where(okT, T, np.nan), p, pid, *SPIKE["T"])
    sS = pq.vertical_spike(np.where(okS, S, np.nan), p, pid, *SPIKE["S"])
    drop("spike", sT, sS)
    okT &= ~sT
    okS &= ~sS
    drop("S_needs_T", np.zeros_like(okT), ~okT)
    okS &= okT                                              # S is computed from C and T

    # bins
    Tb = pq.bin_profiles(pid, z, np.where(okT, T, np.nan), EDGES, npr)
    Sb = pq.bin_profiles(pid, z, np.where(okS, S, np.nan), EDGES, npr)
    pb = pq.bin_profiles(pid, z, np.where(np.isfinite(T) | np.isfinite(S), p, np.nan), EDGES, npr)
    pb = np.where(np.isfinite(pb), pb, gsw.p_from_z(-ZB, 40.0)[None, :])      # empty bins: nominal pressure
    th, sig = _teos(Tb, Sb, pb, prof["lat"], prof["lon"])
    inv = pq.density_inversion(sig, INVERSION)
    n_inv = int(inv.sum())
    Sb[inv] = np.nan
    sig[inv] = np.nan
    stT, stS = pq.stuck_profiles(Tb), pq.stuck_profiles(Sb)
    # profile level
    t, la, lo = prof["t"], prof["lat"], prof["lon"]
    inbox = (lo >= DATA_BOX[0]) & (lo <= DATA_BOX[1]) & (la >= DATA_BOX[2]) & (la <= DATA_BOX[3])
    inper = (t >= T0) & (t <= T1)
    dup = pq.duplicate_profiles(t, la, lo)
    fast = pq.speed_flags(t, la, lo, vmax=3.0) if platform == "glider" else np.zeros(npr, bool)
    posf = o["pos_flag"] if o["pos_flag"].size == npr else np.zeros(npr, bool)
    nT = np.isfinite(Tb).sum(axis=1)
    nS = np.isfinite(Sb).sum(axis=1)
    many_inv = inv.sum(axis=1) > 0.5 * np.maximum(nS + inv.sum(axis=1), 1)
    pos_bad = ~inbox | ~inper | dup | fast | posf
    profT = ~pos_bad & ~stT & (nT >= MIN_BINS)
    profS = profT & ~stS & ~many_inv & (nS >= MIN_BINS)
    th[~profT] = np.nan
    Sb[~profS] = np.nan
    sig[~profS] = np.nan
    # accepted fraction of the raw samples (a sample counts if it passed every point test and its profile is kept)
    accT = float((okT & profT[np.clip(pid, 0, None)] & (pid >= 0)).sum() / max(raw_T.sum(), 1))
    accS = float((okS & profS[np.clip(pid, 0, None)] & (pid >= 0)).sum() / max(raw_S.sum(), 1))
    problems = []
    if (~inper).any():
        problems.append(f"{int((~inper).sum())} profiles outside the period")
    if dup.any():
        problems.append(f"{int(dup.sum())} duplicate profiles")
    if fast.any() or posf.any():
        problems.append(f"{int((fast | posf).sum())} profiles with a bad position (speed > 3 m/s or provider flag)")
    row = dict(record=record, platform=platform, flags=scheme or "none", n_prof=int(npr),
               n_raw_T=int(raw_T.sum()), n_raw_S=int(raw_S.sum()),
               prof_ok_T=int(profT.sum()), prof_ok_S=int(profS.sum()), acc_T=accT, acc_S=accS,
               inversion_bins=n_inv, stuck_T=int(stT.sum()), stuck_S=int(stS.sum()),
               t_first=str(t.min())[:16] if npr else "", t_last=str(t.max())[:16] if npr else "",
               lon_range=f"{np.nanmin(lo):.2f}…{np.nanmax(lo):.2f}" if npr else "",
               lat_range=f"{np.nanmin(la):.2f}…{np.nanmax(la):.2f}" if npr else "",
               z_max=float(np.nanmax(np.where(np.isfinite(th), ZB[None, :], np.nan))) if profT.any() else np.nan,
               problems="; ".join(problems))
    for k, (a, b) in cnt.items():
        row[f"rej_{k}_T"], row[f"rej_{k}_S"] = a, b
    acc = dict(t=t, lat=la, lon=lo, th=th.astype(np.float32), S=Sb.astype(np.float32), sig=sig.astype(np.float32),
               okT=profT, okS=profS,
               _obs_ok=okS & profS[np.clip(pid, 0, None)] & (pid >= 0))    # sample-level acceptance (for figures)
    return row, acc


# ---------------------------------------------------------------------------------------------------- cross-check
def _pair_diffs(a: dict, b: dict, z_min: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(times, median ΔT, median ΔS) over common bins (deeper than ``z_min``) of all profile pairs a − b."""
    ia, ib = pq.pair_profiles(a["t"], a["lat"], a["lon"], b["t"], b["lat"], b["lon"], PAIR_KM, PAIR_DT)
    sel = ZB >= z_min
    if ia.size == 0:
        return np.zeros(0, "datetime64[ns]"), np.zeros(0), np.zeros(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dT = np.nanmedian(a["th"][ia][:, sel] - b["th"][ib][:, sel], axis=1)
        dS = np.nanmedian(a["S"][ia][:, sel] - b["S"][ib][:, sel], axis=1)
    return a["t"][ia], dT, dS


def _row(tg: str, rf: str, t: np.ndarray, dT: np.ndarray, dS: np.ndarray) -> dict:
    bt, bs = pq.bias_drift(t, dT), pq.bias_drift(t, dS)
    span = float(np.ptp(t) / np.timedelta64(1, "D")) if t.size else 0.0
    n_days = int(np.unique(np.asarray(t, "datetime64[D]")[np.isfinite(dS)]).size) if t.size else 0
    return dict(target=tg, reference=rf, n=bt["n"], span_d=span, n_days=n_days, dT=bt["median"], dT_sigma=bt["sigma"],
                dT_drift30=bt["drift_per_30d"], dS=bs["median"], dS_sigma=bs["sigma"],
                dS_drift30=bs["drift_per_30d"], n_S=bs["n"])


def cross_check(acc: dict[str, dict], targets: list[str], references: list[str]) -> list[dict]:
    """Bias and drift of each target record against each reference record, and the bias against all references
    pooled (reference 'ALL'; its drift is not used: pooling references sampled at different times mixes their
    offsets into a spurious trend)."""
    out = []
    for tg in targets:
        pooled = ([], [], [])
        for rf in references:
            if rf == tg:
                continue
            t, dT, dS = _pair_diffs(acc[tg], acc[rf])
            if t.size == 0:
                continue
            for lst, v in zip(pooled, (t, dT, dS)):
                lst.append(v)
            out.append(_row(tg, rf, t, dT, dS))
        if pooled[0]:
            out.append(_row(tg, "ALL", *(np.concatenate(x) for x in pooled)))
    return out


DEEP_THETA, DEEP_ZMIN, DEEP_DRIFT, DEEP_BIAS, DEEP_MIN = 6.0, 150.0, 0.02, 0.03, 20


def deep_theta_s(acc: dict[str, dict]) -> list[dict]:
    """Deep θ–S check of every profiling record: S on θ = 6 °C below 150 m (``profile_qc.salinity_on_theta``), its
    median, Theil–Sen drift over the deployment and the offset from the median of all records."""
    rows = []
    for k, a in acc.items():
        s6 = pq.salinity_on_theta(ZB, a["th"], a["S"], DEEP_THETA, DEEP_ZMIN)
        ok = np.isfinite(s6) & a["okS"]
        if ok.sum() < 3:
            continue
        b = pq.bias_drift(a["t"][ok], s6[ok])
        span = float(np.ptp(a["t"][ok]) / np.timedelta64(1, "D"))
        rows.append(dict(record=k, n=b["n"], span_d=span, S6=b["median"], S6_sigma=b["sigma"],
                         drift_span=b["drift_per_30d"] * span / 30.0 if np.isfinite(b["drift_per_30d"]) else np.nan))
    if rows:
        ens = float(np.median([r["S6"] for r in rows if r["n"] >= DEEP_MIN] or [np.nan]))
        for r in rows:
            r["offset"] = r["S6"] - ens
    return rows


def cross_verdict(rows: list[dict], deep: list[dict], target: str,
                  drift_refs: list[str]) -> tuple[list[str], list[str]]:
    """(fatal T problems, fatal S problems) of a target.

    * bias: pooled median difference to independent profiles beyond ``BIAS_T`` / ``BIAS_S`` (≥ ``CROSS_MIN`` pairs);
    * drift: against one reference of ``drift_refs`` (references that passed the deep θ–S check or carry QARTOD
      flags) with ≥ 30 pairs on ≥ 7 days over ≥ 14 days, beyond ``DRIFT_S`` per 30 days and beyond 3 σ/√n_days over
      the span (σ = 1.48 MAD of the pair differences: at 7 m, 5 km apart, the real variability is 0.1–0.4 PSU);
    * deep θ–S (≥ ``DEEP_MIN`` profiles over ≥ 10 days): drift over the deployment beyond ``DEEP_DRIFT`` or offset from
      the other records beyond ``DEEP_BIAS``."""
    fT, fS = [], []
    for r in rows:
        if r["target"] != target:
            continue
        if r["reference"] == "ALL":
            if r["n"] >= CROSS_MIN and abs(r["dT"]) > BIAS_T:
                fT.append(f"T bias {r['dT']:+.2f} °C vs {r['n']} independent profiles")
            if r["n_S"] >= CROSS_MIN and abs(r["dS"]) > BIAS_S:
                fS.append(f"S bias {r['dS']:+.3f} vs {r['n_S']} independent profiles")
        elif (r["reference"] in drift_refs and r["n_S"] >= 30 and r["span_d"] >= 14.0 and r["n_days"] >= 7
              and np.isfinite(r["dS_drift30"]) and abs(r["dS_drift30"]) > DRIFT_S
              and abs(r["dS_drift30"] * r["span_d"] / 30.0) > 3.0 * r["dS_sigma"] / np.sqrt(r["n_days"])):
            fS.append(f"S drift {r['dS_drift30']:+.3f}/30 d vs {r['reference'].split('-')[0]} ({r['n_S']} pairs)")
    for d in deep:
        if d["record"] != target or d["n"] < DEEP_MIN or d["span_d"] < 10.0:
            continue
        if np.isfinite(d["drift_span"]) and abs(d["drift_span"]) > DEEP_DRIFT:
            fS.append(f"deep θ–S drift {d['drift_span']:+.3f} over {d['span_d']:.0f} d")
        if abs(d["offset"]) > DEEP_BIAS:
            fS.append(f"deep θ–S offset {d['offset']:+.3f}")
    return fT, fS


# ---------------------------------------------------------------------------------------------------- moored CTDs
def qc_fixed_ctd(path: Path, p_tol: float = 3.0) -> tuple[dict, dict]:
    """Moored CTD (NSIF ~7 m or seafloor): provider QARTOD aggregate, gross range, off-station pressure, spikes and
    stuck values on the 15-min grid (``buoy_qc.qc_buoy_series``), gaps."""
    import xarray as xr

    from diagnostics.buoy_qc import qc_buoy_series

    with xr.open_dataset(path) as d:
        t = d["time"].values.astype("datetime64[ns]")
        T, qT = _ooi(d, "sea_water_temperature")
        S, qS = _ooi(d, "sea_water_practical_salinity")
        p, qp = _ooi(d, "sea_water_pressure")
        C, qC = _ooi(d, "sea_water_electrical_conductivity")
        lat, lon = float(d["latitude"].values[0]), float(d["longitude"].values[0])
    off = pq.pressure_off_station(p, p_tol, window=97)              # 97 × 15 min ≈ 1 day running median
    prov = ~(pq.provider_ok(qT, "qartod") & pq.provider_ok(qp, "qartod"))
    provS = ~(pq.provider_ok(qS, "qartod") & pq.provider_ok(qC, "qartod")) | prov
    rT, rS = pq.gross_range(T, *RANGE_T), pq.gross_range(S, *RANGE_S)
    Tm = np.where(prov | off | rT, np.nan, T)
    Sm = np.where(provS | off | rS | ~np.isfinite(Tm), np.nan, S)
    step = np.timedelta64(15, "m")
    qt = qc_buoy_series(t, Tm, RANGE_T, step=step, t_start=t.min(), t_end=t.max() + step, spike_min=1.0,
                        stuck_hours=6.0, min_gap_hours=24.0)
    qs = qc_buoy_series(t, Sm, RANGE_S, step=step, t_start=t.min(), t_end=t.max() + step, spike_min=0.3,
                        stuck_hours=6.0, min_gap_hours=24.0)
    pg = qc_buoy_series(t, np.where(off, np.nan, p), RANGE_P, step=step, t_start=t.min(), t_end=t.max() + step,
                        spike_min=1e9, stuck_hours=1e9)["x"]
    okS = qs["ok"] & qt["ok"]
    n = np.isfinite(T).sum()
    # redeployments (turnarounds): daily-median pressure steps larger than p_tol (information, not a fault)
    import pandas as pd
    dm = pd.Series(np.where(off, np.nan, p), index=t).resample("1D").median().dropna()
    steps = np.flatnonzero(np.abs(np.diff(dm.to_numpy())) > p_tol)
    notes = "; ".join(f"redeployed {str(dm.index[k + 1])[:10]}: p {dm.iloc[k]:.1f} → {dm.iloc[k + 1]:.1f} dbar"
                      for k in steps)
    accT, accS = qt["ok"].sum() / max(n, 1), okS.sum() / max(np.isfinite(S).sum(), 1)
    gaps = [(str(a)[:16], h) for a, h in qt["gaps"]]
    problems = [f"gap {a} ({h:.0f} h)" for a, h in gaps]
    row = dict(record=_record(path), platform="moored_ctd", flags="qartod", n_raw_T=int(n),
               n_raw_S=int(np.isfinite(S).sum()), acc_T=float(min(accT, 1.0)), acc_S=float(min(accS, 1.0)),
               rej_provider_T=int((prov & np.isfinite(T)).sum()), rej_provider_S=int((provS & np.isfinite(S)).sum()),
               rej_offstation=int(off.sum()), rej_spike_T=int(qt["spike"].sum()), rej_spike_S=int(qs["spike"].sum()),
               rej_stuck_T=int(qt["stuck"].sum()), rej_stuck_S=int(qs["stuck"].sum()),
               p_median=float(np.nanmedian(p[~off])), t_first=str(t.min())[:16], t_last=str(t.max())[:16],
               problems="; ".join(problems), notes=notes, lat=lat, lon=lon)
    ser = dict(t=qt["t"], T=np.where(qt["ok"], qt["x"], np.nan), S=np.where(okS, qs["x"], np.nan), p=pg,
               T_raw=qt["x"], S_raw=qs["x"], lat=lat, lon=lon)
    return row, ser


def fixed_as_profiles(ser: dict) -> dict:
    """A moored CTD series as one-bin 'profiles' (15-min samples) for the glider cross-check."""
    import gsw

    k = np.flatnonzero(np.isfinite(ser["T"]))
    n = k.size
    z = -gsw.z_from_p(ser["p"][k], ser["lat"])
    b = np.searchsorted(EDGES, z, side="right") - 1
    th = np.full((n, ZB.size), np.nan, np.float32)
    S = th.copy()
    ok = (b >= 0) & (b < ZB.size) & np.isfinite(z)
    pp = np.where(np.isfinite(ser["p"][k]), ser["p"][k], 0.0)
    tt, _ = _teos(ser["T"][k][:, None], ser["S"][k][:, None], pp[:, None], np.full(n, ser["lat"]),
                  np.full(n, ser["lon"]))
    th[np.flatnonzero(ok), b[ok]] = tt[ok, 0]
    S[np.flatnonzero(ok), b[ok]] = ser["S"][k][ok]
    return dict(t=ser["t"][k], lat=np.full(n, ser["lat"]), lon=np.full(n, ser["lon"]), th=th, S=S)


# ---------------------------------------------------------------------------------------------------- velpt, presf
def qc_velpt(path: Path) -> dict:
    """Point velocity meter: provider flags, |u|, |v| ≤ 1.5 m s⁻¹, |w| ≤ 0.5 m s⁻¹, pressure units against the
    co-located CTD (the files label kPa-like values as dbar).  Not used in the comparison (the ADCPs cover the
    currents)."""
    import xarray as xr

    with xr.open_dataset(path) as d:
        u, qu = _ooi(d, "eastward_sea_water_velocity")
        v, qv = _ooi(d, "northward_sea_water_velocity")
        w, qw = _ooi(d, "upward_sea_water_velocity")
        p, _ = _ooi(d, "sea_water_pressure")
        t = d["time"].values
    bad = ~(pq.provider_ok(qu, "qartod") & pq.provider_ok(qv, "qartod")) | (np.abs(u) > 1.5) | (np.abs(v) > 1.5) \
        | (np.abs(w) > 0.5)
    acc = float(1 - bad.mean())
    pm = float(np.nanmedian(p))
    rec = _record(path)
    nominal = 7.0 if "rid26" in rec else np.nan
    problems = []
    if np.isfinite(nominal) and abs(pm / 10 - nominal) < 3:
        problems.append(f"pressure median {pm:.1f} labelled dbar = {pm / 10:.1f} dbar if kPa (units ×10)")
    elif "mfd35" in rec:
        problems.append(f"pressure median {pm:.0f} labelled dbar; ×0.1 matches the seafloor depth (units ×10)")
    return dict(record=rec, platform="velpt", flags="qartod", n_raw_T=int(np.isfinite(u).sum()), acc_T=acc,
                acc_S=np.nan, t_first=str(t.min())[:16], t_last=str(t.max())[:16],
                rej_w=int((np.abs(w) > 0.5).sum()), problems="; ".join(problems), p_median=pm,
                verdict_T=pq.classify(acc, problems), verdict_S="—", used="no (ADCP covers currents)")


def qc_presf(path: Path, p_tol: float = 3.0) -> dict:
    """Seafloor pressure (tide) recorder: provider flags and off-station pressure.  Not used (η is compared at the
    CO-OPS gauges); its temperature is redundant with the seafloor CTD."""
    import xarray as xr

    with xr.open_dataset(path) as d:
        p, qp = _ooi(d, "sea_water_pressure_at_sea_floor_tide")
        T, qT = _ooi(d, "sea_water_temperature_tide")
        t = d["time"].values.astype("datetime64[ns]")
    off = pq.pressure_off_station(p, p_tol, window=25)              # 25 h running median
    bad = off | ~pq.provider_ok(qp, "qartod")
    acc = float(1 - bad.mean())
    problems = []
    if off.any():
        problems.append(f"{int(off.sum())} h off station (first {str(t[off][0])[:16]})")
    return dict(record=_record(path), platform="presf", flags="qartod", n_raw_T=int(np.isfinite(p).sum()),
                acc_T=acc, acc_S=np.nan, rej_offstation=int(off.sum()),
                rej_provider_T=int((~pq.provider_ok(qp, "qartod")).sum()), p_median=float(np.nanmedian(p[~off])),
                t_first=str(t.min())[:16], t_last=str(t.max())[:16], problems="; ".join(problems),
                verdict_T=pq.classify(acc, problems), verdict_S="—", used="no (η at the CO-OPS gauges)")


# ---------------------------------------------------------------------------------------------------- ADCP
ADCP_E, ADCP_U, ADCP_W, SIDELOBE = 0.1, 1.5, 0.2, 0.06       # m/s, m/s, m/s, 1 − cos 20°


def qc_adcp(path: Path) -> tuple[dict, dict]:
    """ADCP without provider QC (qc_agg = 2 everywhere): side-lobe contamination near the surface (depth <
    0.06 D + 1.5 bin, beam angle 20°; Teledyne RDI 2011 *Acoustic Doppler current profiler principles of
    operation*) [verificar], error velocity |e| > 0.1 m s⁻¹, |u|, |v| > 1.5 m s⁻¹, |w| > 0.2 m s⁻¹, and ensembles with
    more than half of their remaining bins rejected."""
    import xarray as xr

    with xr.open_dataset(path) as d:
        t = d["time"].values.astype("datetime64[ns]")
        z = d["z"].values.astype(float)
        u = d["eastward_sea_water_velocity"].values.astype(float)
        v = d["northward_sea_water_velocity"].values.astype(float)
        w = d["upward_sea_water_velocity"].values.astype(float)
        e = d["velprof_evl"].values.astype(float)
        lat, lon = float(d["latitude"].values[0]), float(d["longitude"].values[0])
    depth = -z
    D = float(np.nanpercentile(depth, 99.5)) + 3.0                       # instrument depth (≈ deepest bin + blank)
    _, inv = np.unique(t, return_inverse=True)
    cell = float(np.median(np.abs(np.diff(depth))[np.diff(inv) == 0])) if depth.size > 1 else 1.0
    side = depth < SIDELOBE * D + 1.5 * cell
    bad_e = ~np.isfinite(e) | (np.abs(e) > ADCP_E)
    bad_u = (np.abs(u) > ADCP_U) | (np.abs(v) > ADCP_U) | ~np.isfinite(u) | ~np.isfinite(v)
    bad_w = np.abs(w) > ADCP_W
    bad = side | bad_e | bad_u | bad_w
    frac = np.bincount(inv, weights=bad & ~side) / np.maximum(np.bincount(inv, weights=~side), 1)
    bad_ens = frac[inv] > 0.5
    bad |= bad_ens
    acc = float((~bad).sum() / max((~side).sum(), 1))
    row = dict(record=_record(path), platform="adcp", flags="none", n_raw_T=int(t.size), acc_T=acc, acc_S=np.nan,
               D_est=D, cell=cell, rej_sidelobe=int(side.sum()), rej_error_vel=int((bad_e & ~side).sum()),
               rej_speed=int((bad_u & ~side).sum()), rej_w=int((bad_w & ~side).sum()),
               rej_ensembles=int(np.unique(inv[bad_ens]).size), t_first=str(t.min())[:16],
               t_last=str(t.max())[:16], lat=lat, lon=lon, problems="")
    ok = ~bad
    ser = dict(t=t[ok], z=depth[ok], u=u[ok], v=v[ok], lat=lat, lon=lon)
    return row, ser


# ---------------------------------------------------------------------------------------------------- eMOLT
EMOLT_TRIM = np.timedelta64(3, "h")


def emolt_sites(path: Path) -> dict[str, dict]:
    """eMOLT records grouped by position (the dataset has no site ID): {site: t, T, depth, lat, lon}."""
    import xarray as xr

    with xr.open_dataset(path) as d:
        t = d["time"].values.astype("datetime64[ns]")
        la, lo = d["latitude"].values.astype(float), d["longitude"].values.astype(float)
        dep, T = d["depth"].values.astype(float), d["sea_water_temperature"].values.astype(float)
    key = np.char.add(np.char.add(np.round(la, 4).astype(str), "_"), np.round(lo, 4).astype(str))
    out = {}
    for k in np.unique(key):
        m = key == k
        o = np.argsort(t[m])
        out[str(k)] = dict(t=t[m][o], T=T[m][o], depth=dep[m][o], lat=float(la[m][0]), lon=float(lo[m][0]))
    return out


def model_site(lat: float, lon: float, hg, H: np.ndarray) -> dict:
    """Nearest wet model T cell to a point: (j, i, distance [km], model depth H [m], inside the cut, Δx [km])."""
    la, lo = hg["lat_t"].values, hg["lon_t"].values
    d = pq.haversine_km(lat, lon, la, lo)
    wet = np.isfinite(H) & (H > 0)
    dd = np.where(wet, d, np.inf)
    j, i = np.unravel_index(np.argmin(dd), dd.shape)
    dx = float(np.sqrt(hg["area_t"].values[j, i]) / 1e3)
    from validation.nwa12_cut import case as cs
    b = cs.CASE.box
    inside = (b[0] <= lon <= b[1]) and (b[2] <= lat <= b[3])
    return dict(j=int(j), i=int(i), dist_km=float(dd[j, i]), H=float(H[j, i]), inside=bool(inside), dx_km=dx,
                nearest_any_km=float(d.min()))


def qc_emolt(sites: dict[str, dict], hg, H: np.ndarray) -> tuple[list[dict], dict]:
    """QC of every eMOLT site plus its representativeness on the model grid."""
    from diagnostics.buoy_qc import qc_buoy_series

    rows, acc = [], {}
    for k, s in sites.items():
        t, T = s["t"], s["T"]
        trim = (t < t.min() + EMOLT_TRIM) | (t > t.max() - EMOLT_TRIM)
        q = qc_buoy_series(t, np.where(trim, np.nan, T), (-2.0, 30.0), t_start=t.min(),
                           t_end=t.max() + np.timedelta64(1, "h"), spike_min=2.0, stuck_hours=48.0,
                           min_gap_hours=24.0)
        n = int(np.isfinite(T).sum())
        a = float(q["ok"].sum() / max(n, 1))
        dvar = float(np.ptp(s["depth"]))
        problems = [f"gap {str(g)[:16]} ({h:.0f} h)" for g, h in q["gaps"]]
        if dvar > 1.0:
            problems.append(f"sensor depth varies by {dvar:.1f} m")
        m = model_site(s["lat"], s["lon"], hg, H)
        dep = float(np.median(s["depth"]))
        why = []
        if not m["inside"]:
            why.append("outside the cut")
        elif m["dist_km"] > 1.5 * m["dx_km"]:
            why.append(f"no wet model cell within 1.5 cells ({m['dist_km']:.0f} km)")
        elif not (0.5 * dep <= m["H"] <= 2.0 * dep):
            why.append(f"model depth {m['H']:.0f} m vs sensor {dep:.1f} m (not within ×2)")
        enters = not why
        verdict = pq.classify(a, problems)
        rows.append(dict(record=f"emolt_{k}", platform="emolt", flags="none", lat=s["lat"], lon=s["lon"],
                         depth=dep, n_raw_T=n, acc_T=a, acc_S=np.nan, rej_spike_T=int(q["spike"].sum()),
                         rej_stuck_T=int(q["stuck"].sum()), rej_trim=int(trim.sum()),
                         t_first=str(t.min())[:16], t_last=str(t.max())[:16], problems="; ".join(problems),
                         verdict_T=verdict, verdict_S="—", model_H=m["H"], model_dist_km=m["dist_km"],
                         used="yes" if enters and verdict != "corrupted" else "no: " + "; ".join(why or [verdict])))
        acc[k] = dict(t=q["t"], T=np.where(q["ok"], q["x"], np.nan), lat=s["lat"], lon=s["lon"], depth=dep,
                      enters=enters and verdict != "corrupted", j=m["j"], i=m["i"])
    return rows, acc


# ---------------------------------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_map(acc: dict, emolt: dict, rows: list[dict], path: Path) -> None:
    plt = _plt()
    from validation.nwa12_cut import case as cs
    hg, H = cs.grid(cs.CASE)
    verdict = {r["record"]: r for r in rows}
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    for a_, zoom in zip(ax, (None, (-71.4, -69.8, 39.2, 40.6))):
        a_.contour(hg["lon_t"].values, hg["lat_t"].values, np.nan_to_num(H), [50, 100, 200, 1000, 3000],
                   colors="0.7", linewidths=0.5)
        b = cs.CASE.box
        a_.plot([b[0], b[1], b[1], b[0], b[0]], [b[2], b[2], b[3], b[3], b[2]], "k--", lw=0.8)
        for n, (k, s) in enumerate(acc.items()):
            r = verdict.get(k, {})
            col = {"usable": "C2", "partial": "C1", "corrupted": "C3"}.get(r.get("verdict_T", ""), "0.5")
            ok = s["okT"]
            a_.plot(s["lon"][ok], s["lat"][ok], ".", ms=2, color=col)
            a_.plot(s["lon"][~ok], s["lat"][~ok], "x", ms=3, color="k", alpha=0.4)
            if zoom is None and ok.any() and r.get("platform") == "glider":
                a_.annotate(k.split("-")[0], (np.nanmedian(s["lon"][ok]), np.nanmedian(s["lat"][ok])), fontsize=6)
        for k, s in emolt.items():
            r = verdict.get(f"emolt_{k}", {})
            a_.plot(s["lon"], s["lat"], "s", mfc="none", ms=7, color="C0" if s["enters"] else "C3")
        if zoom:
            a_.set_xlim(zoom[:2])
            a_.set_ylim(zoom[2:])
        a_.set_title("accepted profiles (green usable, orange partial, red corrupted; x rejected), "
                     "eMOLT □ (blue used, red not)" if zoom is None else "OOI Pioneer array", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_ts(acc: dict, raw: dict, path: Path) -> None:
    """θ–S of accepted bins per platform group, with the rejected raw samples."""
    plt = _plt()
    groups = {"OOI gliders": [k for k in acc if k.startswith("cp_")],
              "other gliders": [k for k in acc if acc[k].get("kind") == "glider" and not k.startswith("cp_")],
              "OOI profilers": [k for k in acc if "wfp" in k]}
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for a_, (g, ks) in zip(ax, groups.items()):
        for k in ks:
            r = raw.get(k)
            if r is not None and r["S"].size:
                a_.plot(r["S"], r["T"], ",", color="C3", alpha=0.3)
            s = acc[k]
            m = np.isfinite(s["S"]) & np.isfinite(s["th"])
            a_.plot(s["S"][m], s["th"][m], ",", color="k", alpha=0.2)
        a_.set_title(f"{g}: accepted bins (black), rejected samples (red)", fontsize=8)
        a_.set_xlabel("S_P")
    ax[0].set_ylabel("θ [°C]")
    ax[0].set_xlim(26, 37.5)
    ax[0].set_ylim(-2, 32)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_cross(cross: list[dict], path: Path) -> None:
    plt = _plt()
    rs = [r for r in cross if r["reference"] == "ALL"]
    if not rs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.35 * len(rs) + 2))
    y = np.arange(len(rs))
    for a_, key, thr, unit in ((ax[0], "dT", BIAS_T, "°C"), (ax[1], "dS", BIAS_S, "")):
        a_.errorbar([r[key] for r in rs], y, xerr=[r[key + "_sigma"] for r in rs], fmt="o", ms=4)
        a_.axvline(-thr, color="C3", lw=0.6)
        a_.axvline(thr, color="C3", lw=0.6)
        a_.axvline(0, color="k", lw=0.5)
        a_.set_yticks(y)
        a_.set_yticklabels([f"{r['target']} (n={r['n']})" for r in rs], fontsize=7)
        a_.set_xlabel(f"median {key} vs independent profiles {unit} (bars: 1.48 MAD)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_moored(series: dict, path: Path) -> None:
    plt = _plt()
    ks = list(series)
    fig, ax = plt.subplots(len(ks), 2, figsize=(14, 1.5 * len(ks) + 1), sharex=True)
    for n, k in enumerate(ks):
        s = series[k]
        for c, v in enumerate(("T", "S")):
            a_ = ax[n, c]
            a_.plot(s["t"], s[v + "_raw"], color="C3", lw=0.5)
            a_.plot(s["t"], s[v], color="k", lw=0.5)
            a_.set_ylabel(k.replace("-03-ctdbp", " ")[:22], fontsize=6)
    ax[0, 0].set_title("θ-sensor T (black accepted, red rejected)", fontsize=8)
    ax[0, 1].set_title("S_P", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_emolt(sites: dict, acc: dict, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(len(sites), 1, figsize=(12, 1.2 * len(sites) + 1), sharex=True)
    for a_, (k, s) in zip(ax, sites.items()):
        a_.plot(s["t"], s["T"], color="C3", lw=0.6)
        a_.plot(acc[k]["t"], acc[k]["T"], color="k" if acc[k]["enters"] else "0.6", lw=0.6)
        a_.set_ylabel(f"{k}\n{acc[k]['depth']:.0f} m", fontsize=6)
    ax[0].set_title("eMOLT bottom T [°C]: black accepted and used, grey accepted but not representable, red rejected",
                    fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- main
def run(data: Path = DATA, out: Path = OUT) -> dict:
    from validation.nwa12_cut import case as cs

    out.mkdir(parents=True, exist_ok=True)
    rows, acc, raw_rej = [], {}, {}
    # profiles: gliders and profiler moorings
    for f in _files(data / "gliders"):
        o = load_glider(f)
        rec = f.stem.replace("-delayed", "")
        dac = np.isfinite(o["flags"]["T"]).any()
        row, a = qc_profiles(o, rec, "glider", "dac" if dac else None)
        if not dac:
            row["flags"] = "none (absent)" if rec.startswith("cp_") else "none (DAC flags all NaN)"
        a["kind"] = "glider"
        rows.append(row)
        acc[rec] = a
        sel = np.isfinite(o["T"]) & np.isfinite(o["S"]) & ~a.pop("_obs_ok")
        rr = np.random.default_rng(0).choice(np.flatnonzero(sel), min(sel.sum(), 20000), replace=False) \
            if sel.any() else np.zeros(0, int)
        raw_rej[rec] = dict(T=o["T"][rr], S=o["S"][rr])
        print(f"{rec}: T {row['acc_T']:.3f} S {row['acc_S']:.3f} prof {row['prof_ok_T']}/{row['n_prof']}", flush=True)
    for f in _files(data / "ooi_pioneer", "*wfp*.nc"):
        o = load_wfp(f)
        rec = _record(f)
        row, a = qc_profiles(o, rec, "profiler", "qartod")
        a.pop("_obs_ok")
        a["kind"] = "profiler"
        rows.append(row)
        acc[rec] = a
        print(f"{rec}: T {row['acc_T']:.3f} S {row['acc_S']:.3f} legs {row['prof_ok_T']}/{row['n_prof']}", flush=True)
    # moored CTDs
    moored = {}
    for f in _files(data / "ooi_pioneer", "*ctdb*.nc"):
        row, ser = qc_fixed_ctd(f)
        rows.append(row)
        moored[_record(f)] = ser
    # cross-checks: deep θ–S first (gliders and profilers reaching θ = 6 °C), then pairs.  Drift references: profiler
    # moorings (QARTOD-flagged) for the gliders; gliders that pass the deep θ–S check for the moored CTDs and the
    # profilers (a moored CTD cannot be checked against itself or against an unverified sensor)
    gl = [k for k in acc if acc[k]["kind"] == "glider"]
    wfp = [k for k in acc if acc[k]["kind"] == "profiler"]
    as_prof = {k: fixed_as_profiles(s) for k, s in moored.items()}
    pool = {**acc, **as_prof}
    deep = deep_theta_s(acc)
    deep_ok = [d["record"] for d in deep if d["n"] >= DEEP_MIN and d["span_d"] >= 10.0
               and not cross_verdict([], deep, d["record"], [])[1]]
    cross = cross_check(pool, gl, wfp + gl)
    cross += cross_check(pool, list(as_prof), gl)
    cross += cross_check(pool, wfp, gl)
    for r in rows:
        if r["platform"] in ("glider", "profiler", "moored_ctd"):
            refs = wfp if r["platform"] == "glider" else deep_ok
            fT, fS = cross_verdict(cross, deep, r["record"], refs)
            probs = [p for p in r["problems"].split("; ") if p]
            r["verdict_T"] = pq.classify(r["acc_T"], probs, fT)
            r["verdict_S"] = pq.classify(r["acc_S"], probs, fS) if r["n_raw_S"] else "—"
            r["fatal"] = "; ".join(fT + fS)
            dd = [d for d in deep if d["record"] == r["record"]]
            if dd:
                r["deep_S6"], r["deep_n"], r["deep_drift"] = dd[0]["S6"], dd[0]["n"], dd[0]["drift_span"]
            if fT and r["record"] in acc:
                acc[r["record"]]["okT"][:] = False
            if fS and r["record"] in acc:
                acc[r["record"]]["okS"][:] = False
                acc[r["record"]]["S"][:] = np.nan
                acc[r["record"]]["sig"][:] = np.nan
            if fS and r["record"] in moored:
                moored[r["record"]]["S"][:] = np.nan
            if fT and r["record"] in moored:
                moored[r["record"]]["T"][:] = np.nan
            r["used"] = "yes" if r["verdict_T"] != "corrupted" else "no"
    for f in _files(data / "ooi_pioneer", "*velpt*.nc"):
        rows.append(qc_velpt(f))
    for f in _files(data / "ooi_pioneer", "*presf*.nc"):
        rows.append(qc_presf(f))
    adcp = {}
    for f in _files(data / "ooi_pioneer", "*adcp*.nc"):
        row, ser = qc_adcp(f)
        row["verdict_T"] = pq.classify(row["acc_T"], [])
        row["verdict_S"] = "—"
        row["used"] = "yes" if row["verdict_T"] != "corrupted" else "no"
        rows.append(row)
        adcp[_record(f)] = ser
    hg, H = cs.grid(cs.CASE)
    sites = emolt_sites(_files(data / "emolt")[0])
    erows, eacc = qc_emolt(sites, hg, H)
    rows += erows

    # files
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "insitu_qc.csv", index=False, float_format="%.5g")
    (out / "insitu_qc.json").write_text(json.dumps(dict(rows=rows, cross=cross, deep=deep), indent=1, default=str))
    save_accepted(out / "insitu_accepted.npz", acc, moored, eacc, adcp)
    plot_map(acc, eacc, rows, out / "insitu_map.png")
    plot_ts(acc, raw_rej, out / "insitu_ts.png")
    plot_cross(cross, out / "insitu_crosscheck.png")
    plot_moored(moored, out / "insitu_moored_ctd.png")
    plot_emolt(sites, eacc, out / "insitu_emolt.png")
    return dict(rows=rows, cross=cross, deep=deep)


def save_accepted(path: Path, acc: dict, moored: dict, emolt: dict, adcp: dict) -> None:
    """Accepted samples as flat arrays (no pickled objects): profiles (θ, S, σ₀ on ZB with record index), moored
    series, eMOLT series and ADCP samples."""
    recs = list(acc)
    keep = [np.flatnonzero(acc[k]["okT"]) for k in recs]
    o = dict(zb=ZB, edges=EDGES, prof_records=np.array(recs),
             prof_rec=np.concatenate([np.full(k.size, n) for n, k in enumerate(keep)]),
             prof_kind=np.array([acc[k]["kind"] for k in recs]),
             prof_t=np.concatenate([acc[r]["t"][k] for r, k in zip(recs, keep)]).astype("datetime64[ns]"),
             prof_lat=np.concatenate([acc[r]["lat"][k] for r, k in zip(recs, keep)]),
             prof_lon=np.concatenate([acc[r]["lon"][k] for r, k in zip(recs, keep)]),
             prof_th=np.concatenate([acc[r]["th"][k] for r, k in zip(recs, keep)]),
             prof_S=np.concatenate([acc[r]["S"][k] for r, k in zip(recs, keep)]),
             prof_sig=np.concatenate([acc[r]["sig"][k] for r, k in zip(recs, keep)]))
    mk = list(moored)
    o["moored_records"] = np.array(mk)
    for n, k in enumerate(mk):
        s = moored[k]
        o[f"moored{n}_t"], o[f"moored{n}_T"], o[f"moored{n}_S"], o[f"moored{n}_p"] = s["t"], s["T"], s["S"], s["p"]
        o[f"moored{n}_pos"] = np.array([s["lat"], s["lon"]])
    ek = list(emolt)
    o["emolt_sites"] = np.array(ek)
    for n, k in enumerate(ek):
        s = emolt[k]
        o[f"emolt{n}_t"], o[f"emolt{n}_T"] = s["t"], s["T"]
        o[f"emolt{n}_meta"] = np.array([s["lat"], s["lon"], s["depth"], float(s["enters"]), s["j"], s["i"]])
    ak = list(adcp)
    o["adcp_records"] = np.array(ak)
    for n, k in enumerate(ak):
        s = adcp[k]
        for v in ("t", "z", "u", "v"):
            o[f"adcp{n}_{v}"] = s[v]
        o[f"adcp{n}_pos"] = np.array([s["lat"], s["lon"]])
    np.savez_compressed(path, **o)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    res = run(a.data, a.out)
    import pandas as pd
    df = pd.DataFrame(res["rows"])
    cols = [c for c in ("record", "platform", "flags", "acc_T", "acc_S", "verdict_T", "verdict_S", "used", "fatal",
                        "problems") if c in df]
    with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 200):
        print(df[cols].to_string())
        print(pd.DataFrame(res["cross"]).round(3).to_string())
        print(pd.DataFrame(res["deep"]).round(4).to_string())


if __name__ == "__main__":
    main()
