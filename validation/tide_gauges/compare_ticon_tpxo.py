#!/usr/bin/env python
"""
TPXO10-atlas-v2 (as read and interpolated by pyOCEAN) vs TICON-3 tide gauges.

For every coastal TICON-3 record in the Mexican Pacific (incl. the Gulf of
California) and in the Gulf of Mexico, the 8 HYCOM constituents of TPXO10
are interpolated to the gauge (land-aware bilinear; if the gauge falls among
land nodes, the nearest ocean node within 3 atlas cells is used), and the
record is diagnosed with diagnostics.tide_gauges.diagnose_station:

    timezone     whole-hour clock offset (local time logged as UTC, DST)
    units        amplitude off by a power of ten
    inverted     sign-flipped record
    lag_damping  amplitude reduction + lag: lagoon/harbour filtering or atlas
                 resolution (physical; a slow clock cannot be excluded)
    misfit       large residual without a simple signature

Records of the same site (within 0.02°) are also cross-checked against each
other, independently of TPXO.

Outputs (validation/tide_gauges/out/):
    ticon_vs_tpxo_<region>.csv   one row per record
    summary printed to stdout

Data: TICON-3 (Hart-Davis et al. 2022, CC-BY-4.0, doi:10.1594/PANGAEA.951610)
in ~/pyOCEAN_refs/sealevel/ticon3; TPXO10-atlas-v2 in
~/pyOCEAN_refs/TPXO10_atlas_v2_nc (academic licence, read in place).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from diagnostics.tide_gauges import compare_with_siblings, diagnose_station, load_ticon  # noqa: E402
from forcing.sources.tpxo import read_tpxo  # noqa: E402
from forcing.tides import CONSTITUENTS, _interp_complex  # noqa: E402

REFS = Path.home() / "pyOCEAN_refs"
TICON = REFS / "sealevel" / "ticon3" / "TICON_3.txt"
ATLAS = REFS / "TPXO10_atlas_v2_nc"
OUT = Path(__file__).parent / "out"
REGIONS = {
    # Pacific side of the continental divide: south of 18 °N (Isthmus of
    # Tehuantepec and the southern Pacific coast) or west of 100 °W.
    "pacific_mx": dict(lon=(-118.0, -92.0), lat=(14.0, 32.6),
                       keep=lambda lat, lon: (lat < 18.0) | (lon < -100.0)),
    "gulf_of_mexico": dict(lon=(-98.5, -80.0), lat=(18.0, 31.0),
                           keep=lambda lat, lon: lon > -98.5),
}
CONS = list(CONSTITUENTS)


TPXO_MIN_DEPTH = 5.0   # m; TPXO10 has spurious metre-scale amplitudes at ~2 m interior nodes (Florida Bay)


def tpxo_depth(grid: xr.Dataset, lat: float, lon: float) -> float:
    """
    Mean TPXO depth hz [m] of the ocean nodes around the gauge.

    Uses the 3 × 3 block of z nodes nearest to the gauge (the bilinear
    stencil plus the fallback neighbourhood), ignoring land (hz = 0);
    0 if all are land.
    """
    i = int(np.argmin(np.abs(((grid["lon_z"].values - lon + 180.0) % 360.0) - 180.0)))
    j = int(np.argmin(np.abs(grid["lat_z"].values - lat)))
    block = grid["hz"].isel(nx=slice(max(i - 1, 0), i + 2), ny=slice(max(j - 1, 0), j + 2)).values
    wet = block[block > 0]
    return float(wet.mean()) if wet.size else 0.0


def verdict(flags: str, sibling_flags: str, depth: float, rss: float, site_min_rss: float) -> str:
    """
    Combine the TPXO comparison, the same-site comparison and the atlas depth.

    A record that disagrees with its siblings by an instrumental signature is
    a confirmed defect only if it is also the worse of them against TPXO
    (with two records, the sibling test alone cannot tell which one is bad).
    """
    instrumental = ("timezone", "units", "inverted")
    if any(f in sibling_flags for f in instrumental) and rss > 2.0 * site_min_rss:
        return "gauge_defect_confirmed"
    if depth < TPXO_MIN_DEPTH or flags == "no_tpxo_ocean_node":
        return "tpxo_unusable"
    if any(f in flags for f in instrumental):
        return "site_or_model" if sibling_flags == "consistent" else "possible_gauge_defect"
    if flags == "ok":
        return "ok"
    return "lag_damping" if flags == "lag_damping" else "misfit"


def tpxo_at(tc, lat: float, lon: float) -> tuple[np.ndarray, float]:
    """TPXO constants at a gauge; fallback to the nearest ocean node (distance in km)."""
    h = _interp_complex(tc.h, "lat", "lon", np.array([lat]), np.array([lon]))[0]
    if np.all(np.isfinite(h)):
        return h, 0.0
    j0 = int(np.argmin(np.abs(tc.lat.values - lat)))
    i0 = int(np.argmin(np.abs(tc.lon.values - lon)))
    best, dist = None, np.inf
    for dj in range(-3, 4):
        for di in range(-3, 4):
            j, i = j0 + dj, i0 + di
            if 0 <= j < tc.lat.size and 0 <= i < tc.lon.size:
                v = tc.h.values[:, j, i]
                if np.all(np.isfinite(v)):
                    d = 111.2 * np.hypot(tc.lat.values[j] - lat, (tc.lon.values[i] - lon) * np.cos(np.deg2rad(lat)))
                    if d < dist:
                        best, dist = v, d
    return (best if best is not None else np.full(len(CONS), np.nan + 0j)), dist


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for region, box in REGIONS.items():
        st = load_ticon(TICON, box["lon"], box["lat"], CONS)
        st = st[box["keep"](st["lat"], st["lon"])].reset_index(drop=True)
        tc = read_tpxo([ATLAS / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in CONS], None,
                       ATLAS / "grid_tpxo10atlas_v2.nc",
                       lon_range=(box["lon"][0] - 0.5, box["lon"][1] + 0.5),
                       lat_range=(box["lat"][0] - 0.5, box["lat"][1] + 0.5))
        grid = xr.open_dataset(ATLAS / "grid_tpxo10atlas_v2.nc")
        rows = []
        for _, r in st.iterrows():
            h_mod, dist = tpxo_at(tc, r["lat"], r["lon"])
            if not np.all(np.isfinite(h_mod)):
                rows.append(dict(lat=r["lat"], lon=r["lon"], start=r["start"].date(), end=r["end"].date(),
                                 flags="no_tpxo_ocean_node"))
                continue
            d = diagnose_station(r["hc"], h_mod, CONS)
            rows.append(dict(
                lat=r["lat"], lon=r["lon"], start=r["start"].date(), end=r["end"].date(), miss=r["miss"],
                src=r["src"], node_km=round(dist, 1), tpxo_depth_m=tpxo_depth(grid, r["lat"], r["lon"]),
                m2_obs_cm=100 * abs(r["hc"][0]), m2_tpxo_cm=100 * abs(h_mod[0]),
                m2_dphase=d["d_phase"][0], k1_dphase=d["d_phase"][2],
                rss_cm=100 * d["rss"], rel_misfit=d["rel_misfit"],
                tau_min=d["tau_minutes"], gain=d["gain"], rss_corr_cm=100 * d["rss_corrected"],
                flags=",".join(d["flags"]) or "ok"))
        df = pd.DataFrame(rows)

        # Same-site records (within 0.02°) compared with each other, independent of TPXO
        df["site"] = -1
        site = 0
        for i in range(len(df)):
            if df.at[i, "site"] >= 0:
                continue
            near = (np.hypot(df["lat"] - df.at[i, "lat"], df["lon"] - df.at[i, "lon"]) < 0.02) & (df["site"] < 0)
            df.loc[near, "site"] = site
            site += 1
        df["sibling_flags"] = ""
        df["sib_tau_min"] = np.nan
        df["sib_gain"] = np.nan
        for _, idx in df.groupby("site").groups.items():
            idx = list(idx)
            res = compare_with_siblings([st.at[i, "hc"] for i in idx], CONS)
            for i, d in zip(idx, res):
                if d is not None:
                    df.at[i, "sibling_flags"] = ",".join(f for f in d["flags"] if f != "misfit") or "consistent"
                    df.at[i, "sib_tau_min"], df.at[i, "sib_gain"] = d["tau_minutes"], d["gain"]
        df["tpxo_depth_m"] = df["tpxo_depth_m"].fillna(0.0)
        site_min = df.groupby("site")["rss_cm"].transform("min")
        df["verdict"] = [verdict(f, sf, dep, r, m) for f, sf, dep, r, m in
                         zip(df["flags"], df["sibling_flags"], df["tpxo_depth_m"], df["rss_cm"], site_min)]
        df["mexico"] = (df["lat"] < 25.9) & (df["lon"] < -86.5) if region == "gulf_of_mexico" else True
        df.to_csv(OUT / f"ticon_vs_tpxo_{region}.csv", index=False, float_format="%.4g")
        grid.close()
        cols = ["lat", "lon", "start", "end", "m2_obs_cm", "m2_tpxo_cm", "m2_dphase", "rss_cm",
                "tau_min", "gain", "flags", "sibling_flags", "tpxo_depth_m"]
        mx = df[df["mexico"]]
        print(f"\n=== {region}: Mexican records {len(mx)}; verdicts:",
              dict(mx["verdict"].value_counts()))
        usable = mx[mx["verdict"] == "ok"]
        if len(usable):
            print(f"  ok records: median RSS {usable['rss_cm'].median():.1f} cm, "
                  f"median |ΔG_M2| {usable['m2_dphase'].abs().median():.1f}°, "
                  f"median |ΔA_M2| {np.median(np.abs(usable['m2_obs_cm'] - usable['m2_tpxo_cm'])):.1f} cm")
        for v in ("gauge_defect_confirmed", "possible_gauge_defect", "site_or_model", "lag_damping", "misfit",
                  "tpxo_unusable"):
            sub = mx[mx["verdict"] == v]
            if len(sub):
                print(f"  --- {v} ({len(sub)})")
                print(sub[cols].to_string(index=False))
        if region == "gulf_of_mexico":
            rest = df[~df["mexico"]]
            print(f"\n  (non-Mexican Gulf records: {len(rest)}; verdicts {dict(rest['verdict'].value_counts())})")
        continue

        ok = df["flags"] == "ok"
        print(f"\n=== {region}: {len(df)} coastal records at {df['site'].nunique()} sites")
        print("  vs TPXO   ok: %d   " % ok.sum() + "   ".join(
            f"{f}: {df['flags'].str.contains(f).sum()}"
            for f in ("timezone", "units", "inverted", "lag_damping", "misfit", "no_tpxo")))
        good = df[ok]
        if len(good):
            print(f"  records without flags: median RSS {good['rss_cm'].median():.1f} cm, "
                  f"median |ΔG_M2| {good['m2_dphase'].abs().median():.1f}°")
        instr = df[df["flags"].str.contains("timezone|units|inverted") |
                   df["sibling_flags"].str.contains("timezone|units|inverted")]
        if len(instr):
            print("  GAUGE DEFECTS (vs TPXO and/or vs same-site records):")
            print(instr[["lat", "lon", "start", "end", "m2_obs_cm", "m2_tpxo_cm", "tau_min", "gain", "flags",
                         "sib_tau_min", "sib_gain", "sibling_flags"]].to_string(index=False))
        lag = df[df["flags"] == "lag_damping"]
        if len(lag):
            print(f"  lag_damping (probably lagoon/harbour or atlas resolution): {len(lag)} records, "
                  f"median gain {lag['gain'].median():.2f}, median lag {lag['tau_min'].median():.0f} min")
            consistent = lag[lag["sibling_flags"] == "consistent"]
            print(f"    of which {len(consistent)} agree with other records of the same site "
                  "(⇒ model/local, not the gauge)")
        mis = df[df["flags"] == "misfit"].sort_values("rel_misfit", ascending=False)
        if len(mis):
            print(f"  misfit (largest {min(12, len(mis))} of {len(mis)}):")
            print(mis.head(12)[["lat", "lon", "start", "end", "m2_obs_cm", "m2_tpxo_cm", "m2_dphase", "rss_cm",
                                "rel_misfit", "node_km"]].to_string(index=False))


if __name__ == "__main__":
    main()
