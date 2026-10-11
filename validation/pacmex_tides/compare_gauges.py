#!/usr/bin/env python
"""
Compare the pyOCEAN PAC_MEX tidal run with tide gauges (TICON-3) and TPXO10.

For every TICON-3 coastal record in the box (records flagged as gauge
defects in validation/tide_gauges are excluded), the model η series at the
nearest wet cell is analysed after the spin-up (``harmonic_analysis``; P1, K2
inferred from K1, S2 with TPXO ratios at the gauge) and compared — like TPXO —
with the gauge constants: vector difference per constituent and
RSS = √(½ Σ |Δh|²).  Maps of the M2 constants of the run are compared with
TPXO10 over the whole subdomain.

Usage:  python compare_gauges.py [tag ...]      (default: ref nosal)
Output: out/gauges_<tag>.csv, out/summary.txt, out/m2_map_<tag>.png, out/rss_by_gauge.png
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
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
from diagnostics.sealevel_qc import harmonic_analysis  # noqa: E402
from diagnostics.tide_gauges import load_ticon  # noqa: E402
from forcing.sources.tpxo import read_tpxo  # noqa: E402
from forcing.tides import CONSTITUENTS, _interp_complex  # noqa: E402

OUT = Path(__file__).parent / "out"
CONS = list(CONSTITUENTS)
VERDICTS = ROOT / "validation" / "tide_gauges" / "out"


def rss(d: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.nansum(np.abs(d) ** 2)))


def analyse(tag: str, tc, ticon: pd.DataFrame, bad: pd.DataFrame) -> pd.DataFrame:
    run = np.load(OUT / f"run_{tag}.npz")
    meta = json.loads(str(run["meta"]))
    t = run["gauge_t"].astype("datetime64[ns]")
    keep = t >= t[0] + np.timedelta64(int(meta["spinup"] * 86400), "s")
    rows = []
    for key, (j, i), series in zip(run["gauge_keys"], run["gauge_ij"], run["gauge_eta"].T):
        lat, lon, year = key.split("_")
        lat, lon = float(lat), float(lon)
        rec = ticon[(np.abs(ticon["lat"] - lat) < 1e-3) & (np.abs(ticon["lon"] - lon) < 1e-3)
                    & (ticon["start"].dt.year == int(year))]
        if rec.empty:
            continue
        r = rec.iloc[0]
        if not bad[(np.abs(bad["lat"] - lat) < 0.01) & (np.abs(bad["lon"] - lon) < 0.01)
                   & (bad["start"] == str(r["start"].date()))].empty:
            continue                                          # defective gauge record
        h_tpxo, _ = tpxo_at(tc, lat, lon)
        inf = {"p1": ("k1", h_tpxo[CONS.index("p1")] / h_tpxo[CONS.index("k1")]),
               "k2": ("s2", h_tpxo[CONS.index("k2")] / h_tpxo[CONS.index("s2")])}
        h_mod = harmonic_analysis(t[keep], series[keep].astype(float), CONS, inference=inf)["hc"]
        obs = r["hc"]
        m2 = CONS.index("m2")
        rows.append(dict(
            lat=lat, lon=lon, period=f"{r['start'].year}-{r['end'].year}", cell=f"{j},{i}",
            m2_obs_cm=100 * abs(obs[m2]), m2_mod_cm=100 * abs(h_mod[m2]), m2_tpxo_cm=100 * abs(h_tpxo[m2]),
            m2_dphase_mod=float((np.rad2deg(np.angle(h_mod[m2]) - np.angle(obs[m2])) + 180) % 360 - 180) * -1,
            m2_dphase_tpxo=float((np.rad2deg(np.angle(h_tpxo[m2]) - np.angle(obs[m2])) + 180) % 360 - 180) * -1,
            k1_obs_cm=100 * abs(obs[CONS.index("k1")]), k1_mod_cm=100 * abs(h_mod[CONS.index("k1")]),
            rss_mod_cm=100 * rss(h_mod - obs), rss_tpxo_cm=100 * rss(h_tpxo - obs),
            rss_mod_vs_tpxo_cm=100 * rss(h_mod - h_tpxo)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"gauges_{tag}.csv", index=False, float_format="%.3f")
    return df


def m2_map(tag: str, tc) -> str:
    run = np.load(OUT / f"run_{tag}.npz")
    k = list(run["ha_cons"]).index("m2")
    hc = run["hc_map"][..., k]
    lon, lat = run["lon_t"], run["lat_t"]
    wet = np.isfinite(hc)
    tp = np.full(hc.shape, np.nan + 0j)
    tp[wet] = _interp_complex(tc.h.sel(constituent=["m2"]), "lat", "lon", lat[wet], lon[wet])[:, 0]
    deep = wet & (run["depth"] > 1000) & np.isfinite(tp)
    d = hc - tp
    stats = (f"{tag}: M2 vs TPXO10 over depth > 1000 m: RMS |Δh| = {100 * np.sqrt(np.nanmean(np.abs(d[deep]) ** 2)):.2f} cm, "
             f"amp ratio median {np.nanmedian(np.abs(hc[deep]) / np.abs(tp[deep])):.3f}, "
             f"phase diff median {np.nanmedian(np.rad2deg(np.angle(tp[deep] / hc[deep]))):+.1f}°")
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    s = (slice(None, None, 2), slice(None, None, 2))
    for a, f, title, kw in ((ax[0], 100 * np.abs(hc), f"pyOCEAN M2 amplitude [cm] ({tag})", dict(vmin=0, vmax=150)),
                            (ax[1], 100 * np.abs(tp), "TPXO10 M2 amplitude [cm]", dict(vmin=0, vmax=150)),
                            (ax[2], 100 * np.abs(d), "|h_pyOCEAN − h_TPXO| [cm]", dict(vmin=0, vmax=10))):
        c = a.pcolormesh(lon[s], lat[s], f[s], shading="auto", **kw)
        fig.colorbar(c, ax=a, shrink=0.8)
        ph = np.rad2deg(-np.angle(hc if a is not ax[1] else tp)) % 360
        if a is not ax[2]:
            a.contour(lon[s], lat[s], np.where(wet, ph, np.nan)[s], levels=np.arange(0, 360, 30),
                      colors="w", linewidths=0.5)
        a.set_title(title, fontsize=9)
        a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / f"m2_map_{tag}.png", dpi=100)
    plt.close(fig)
    return stats


def main() -> None:
    tags = sys.argv[1:] or ["ref", "nosal"]
    box_lon, box_lat = (-118.5, -91.5), (11.5, 32.5)
    tc = read_tpxo([ATLAS / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in CONS], None,
                   ATLAS / "grid_tpxo10atlas_v2.nc", lon_range=box_lon, lat_range=box_lat)
    ticon = load_ticon(TICON, box_lon, box_lat, CONS)
    bad = pd.concat([pd.read_csv(VERDICTS / f"ticon_vs_tpxo_{r}.csv") for r in ("pacific_mx", "gulf_of_mexico")])
    bad = bad[bad["verdict"].isin(["gauge_defect_confirmed", "possible_gauge_defect"])]
    lines, frames = [], {}
    for tag in tags:
        if not (OUT / f"run_{tag}.npz").exists():
            continue
        df = analyse(tag, tc, ticon, bad)
        frames[tag] = df
        lines.append(f"{tag}: {len(df)} gauge records; median RSS model {df.rss_mod_cm.median():.2f} cm, "
                     f"TPXO {df.rss_tpxo_cm.median():.2f} cm; model better at "
                     f"{(df.rss_mod_cm < df.rss_tpxo_cm).sum()}/{len(df)}")
        lines.append(m2_map(tag, tc))
    if "ref" in frames:
        df = frames["ref"].sort_values("lat")
        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(len(df))
        ax.bar(x - 0.27, df.rss_mod_cm, 0.27, label="pyOCEAN (SAL)")
        if "nosal" in frames:
            ax.bar(x, frames["nosal"].set_index(["lat", "lon", "period"]).loc[
                list(zip(df.lat, df.lon, df.period)), "rss_mod_cm"].values, 0.27, label="pyOCEAN (no SAL)")
        ax.bar(x + 0.27, df.rss_tpxo_cm, 0.27, label="TPXO10")
        ax.set_xticks(x, [f"{la:.2f},{lo:.2f}\n{p}" for la, lo, p in zip(df.lat, df.lon, df.period)],
                      rotation=90, fontsize=6)
        ax.set_ylabel("RSS vs gauge [cm]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "rss_by_gauge.png", dpi=110)
        plt.close(fig)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
