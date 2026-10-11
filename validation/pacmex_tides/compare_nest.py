#!/usr/bin/env python
"""
Resolution × nonlinearity × boundary source in the northern Gulf of California.

Runs (out/run_<tag>.npz, same forcing, JSL 0.5 + HYCOM-IOA cb, A_h = 100 m²/s,
2026-08-01 + 33 days):
    p25_lin, p25_nl            1/25° PAC_MEX subdomain, TPXO boundary
    n100_lin_tpxo, n100_nl_tpxo  1/100° GEBCO nest, TPXO boundary
    n100_lin_par, n100_nl_par    1/100° nest, boundary from the 1/25° parent (harmonic)

For every TICON-3 record inside the nest box: M2 amplitude/phase and the
8-constituent RSS against the gauge, from each run's gauge series
(harmonic_analysis, P1/K2 inferred with TPXO ratios), plus TPXO10.

Usage:  python validation/pacmex_tides/compare_nest.py
Output: out/nest_gauges.csv, out/nest_summary.txt, out/nest_m2_maps.png
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
TAGS = ["p25_lin", "p25_nl", "n100_lin_tpxo", "n100_nl_tpxo", "n100_lin_par", "n100_nl_par"]
NEST = dict(lon=(-115.3, -108.7), lat=(25.2, 31.95))


def rss(d: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.nansum(np.abs(d) ** 2)))


def gauge_constants(tag: str, tc) -> dict:
    """{(lat, lon): hc (8,)} from the gauge series of a run, inside the nest box."""
    r = np.load(OUT / f"run_{tag}.npz")
    meta = json.loads(str(r["meta"]))
    t = r["gauge_t"].astype("datetime64[ns]")
    keep = t >= t[0] + np.timedelta64(int(meta["spinup"] * 86400), "s")
    out = {}
    for key, series in zip(r["gauge_keys"], r["gauge_eta"].T):
        lat, lon, _ = key.split("_")
        lat, lon = float(lat), float(lon)
        if not (NEST["lon"][0] <= lon <= NEST["lon"][1] and NEST["lat"][0] <= lat <= NEST["lat"][1]):
            continue
        h_tp, _ = tpxo_at(tc, lat, lon)
        inf = {"p1": ("k1", h_tp[CONS.index("p1")] / h_tp[CONS.index("k1")]),
               "k2": ("s2", h_tp[CONS.index("k2")] / h_tp[CONS.index("s2")])}
        out[(round(lat, 3), round(lon, 3))] = harmonic_analysis(t[keep], series[keep].astype(float), CONS,
                                                                 inference=inf)["hc"]
    return out


def main() -> None:
    tags = [t for t in TAGS if (OUT / f"run_{t}.npz").exists()]
    tc = read_tpxo([ATLAS / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in CONS], None, ATLAS / "grid_tpxo10atlas_v2.nc",
                   lon_range=(NEST["lon"][0] - 0.5, NEST["lon"][1] + 0.5),
                   lat_range=(NEST["lat"][0] - 0.5, NEST["lat"][1] + 0.5))
    ticon = load_ticon(TICON, NEST["lon"], NEST["lat"], CONS)
    runs = {t: gauge_constants(t, tc) for t in tags}
    common = set.intersection(*[set(v) for v in runs.values()]) if runs else set()
    m2 = CONS.index("m2")
    rows = []
    for (lat, lon) in sorted(common):
        rec = ticon[(np.abs(ticon.lat - lat) < 1e-3) & (np.abs(ticon.lon - lon) < 1e-3)]
        if rec.empty:
            continue
        obs = rec.sort_values("end").iloc[-1]["hc"]
        h_tp, _ = tpxo_at(tc, lat, lon)
        row = dict(lat=lat, lon=lon, m2_obs_cm=100 * abs(obs[m2]), m2_obs_G=np.rad2deg(-np.angle(obs[m2])) % 360,
                   tpxo_rss_cm=100 * rss(h_tp - obs), tpxo_m2_cm=100 * abs(h_tp[m2]))
        for t in tags:
            h = runs[t][(lat, lon)]
            row |= {f"{t}_rss_cm": 100 * rss(h - obs), f"{t}_m2_cm": 100 * abs(h[m2]),
                    f"{t}_m2_dG": float((np.rad2deg(np.angle(obs[m2]) - np.angle(h[m2])) + 180) % 360 - 180)}
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("lat")
    df.to_csv(OUT / "nest_gauges.csv", index=False, float_format="%.2f")
    lines = [f"{len(df)} TICON records inside the nest (common to all runs): "
             + ", ".join(f"{a:.2f}N" for a in df.lat)]
    for t in tags + ["tpxo"]:
        lines.append(f"{t:15s} median RSS {df[f'{t}_rss_cm'].median():5.2f} cm   mean {df[f'{t}_rss_cm'].mean():5.2f} cm")
    lines.append("M2 amplitude [cm] (obs | " + " | ".join(tags) + " | tpxo):")
    for _, r in df.iterrows():
        lines.append(f"  {r.lat:6.2f}N {r.lon:8.2f}: {r.m2_obs_cm:5.1f} | "
                     + " | ".join(f"{r[f'{t}_m2_cm']:5.1f} ({r[f'{t}_m2_dG']:+4.0f}°)" for t in tags)
                     + f" | {r.tpxo_m2_cm:5.1f}")
    (OUT / "nest_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # M2 amplitude maps vs TPXO in the nest box
    fig, ax = plt.subplots(2, (len(tags) + 2) // 2, figsize=(5 * ((len(tags) + 2) // 2), 9))
    ax = ax.ravel()
    for a, t in zip(ax, tags):
        r = np.load(OUT / f"run_{t}.npz")
        k = list(r["ha_cons"]).index("m2")
        lon, lat, hc = r["lon_t"], r["lat_t"], r["hc_map"][..., k]
        m = (lon >= NEST["lon"][0]) & (lon <= NEST["lon"][1]) & (lat >= NEST["lat"][0]) & (lat <= NEST["lat"][1])
        jj, ii = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        s = (slice(jj[0], jj[-1] + 1), slice(ii[0], ii[-1] + 1))
        c = a.pcolormesh(lon[s], lat[s], 100 * np.abs(hc[s]), vmin=0, vmax=160, shading="auto")
        a.contour(lon[s], lat[s], np.rad2deg(-np.angle(hc[s])) % 360, levels=np.arange(0, 360, 30),
                  colors="w", linewidths=0.4)
        a.plot(df.lon, df.lat, "r.", ms=4)
        a.set_title(t, fontsize=9)
        a.set_aspect("equal")
    fig.colorbar(c, ax=ax.tolist(), shrink=0.6, label="M2 amplitude [cm]")
    fig.savefig(OUT / "nest_m2_maps.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
