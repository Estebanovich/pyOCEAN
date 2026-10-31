"""
Compare the ERA5 column of pyOCEAN, MITgcm (LY04) and CROCO (COARE 3.0): N2b task 7, spec criterion 5.

    python -m validation.bulk_column.compare [--days 45] [--figs]

Reads RUNS/{mitgcm,croco,pyocean}_<days>d, brings every model to one form (``read_*``) and prints

1. the mean of each surface flux over the run per model and the relative difference of pyOCEAN from its reference
   ('ly04' and 'fms' against MITgcm, 'coare' and 'coare_croco' against CROCO);
2. the SST at the end (last snapshot and last-day mean), its RMS difference over the run, the mixed-layer depth;
3. the split of each flux difference (``split``): the same pyOCEAN bulk function evaluated offline on the
   reference model's own SST and surface current ("formula at the reference state") against the reference's flux
   (the formula/implementation part), and on pyOCEAN's own state (the state/feedback part); the offline evaluation
   on pyOCEAN's own state against its online hourly mean measures the error of the offline approximation (hourly
   SST and current at mid-hour = mean of the two snapshots, atmosphere at mid-hour).

Criterion (spec): mean fluxes within 2 % and final SST within 0.2 °C, or the difference explained and classified.

Common form (``read_*``): t_snap [h], T, S (nt, N) bed first, u_s, v_s (top-layer current), sst (top layer, 0.5 m),
hbl [m] (hourly snapshot or mean), mld [m]; t_mean [h] (end of each hour) and hourly means q_sw, q_lw, q_sens, q_lat
[W m⁻², + into the ocean], tau_x, tau_y [N m⁻²], E, P [kg m⁻² s⁻¹] (mass fluxes: each model's own volume
convention converted back, module docstring of core/components/surface_fluxes.py).

MLD: depth where ρ exceeds the top-layer ρ by 0.03 kg m⁻³ (shared linear EOS; de Boyer Montégut et al. 2004
criterion with the top layer as reference, the shelf is 36 m deep), linear between layer centres.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

from validation.bulk_column import cases as C

FLUX_KEYS = ("q_sw", "q_lw", "q_sens", "q_lat", "q_net", "tau_x", "tau_y", "tau", "E", "P")
LABEL = {"q_sw": "Q_sw", "q_lw": "Q_lw", "q_sens": "Q_sens", "q_lat": "Q_lat", "q_net": "Q_net",
         "tau_x": "τ_x", "tau_y": "τ_y", "tau": "|τ|", "E": "E", "P": "P"}
UNIT = {k: "W m⁻²" for k in ("q_sw", "q_lw", "q_sens", "q_lat", "q_net")}
UNIT.update(tau_x="N m⁻²", tau_y="N m⁻²", tau="N m⁻²", E="mm d⁻¹", P="mm d⁻¹")
REFERENCE = {"ly04": "mitgcm", "fms": "mitgcm", "coare": "croco", "coare_croco": "croco",
             "flux_mitgcm": "mitgcm", "flux_croco": "croco", "flux_croco_bg": "croco", "coare_bg": "croco"}
NAMES = {"mitgcm": "MITgcm exf LY04", "ly04": "pyOCEAN 'ncar_ly04'", "fms": "pyOCEAN 'ncar' (FMS)",
         "croco": "CROCO COARE 3.0", "coare": "pyOCEAN 'coare'", "coare_croco": "pyOCEAN 'coare', CROCO coupling",
         "flux_mitgcm": "pyOCEAN, MITgcm fluxes", "flux_croco": "pyOCEAN, CROCO fluxes",
         "flux_croco_bg": "pyOCEAN, CROCO fluxes + CROCO bg", "coare_bg": "pyOCEAN 'coare' + CROCO bg"}
ORDER = ("mitgcm", "ly04", "croco", "coare", "coare_croco", "fms")
DIAG = ("flux_mitgcm", "flux_croco", "flux_croco_bg", "coare_bg")
MM_DAY = 86400.0                         # kg m⁻² s⁻¹ → mm d⁻¹ of fresh water (ρ_fw = 1000)
CROCO_I = 3                              # an interior ρ point of the 7 × 7 CROCO arrays


def mld(T: np.ndarray, S: np.ndarray, drho: float = 0.03) -> np.ndarray:
    """Mixed-layer depth [m] per snapshot from (nt, N) bed-first profiles (module docstring)."""
    rho = C.rho_linear(T, S)[:, ::-1]                  # top first
    d = C.layer_depths()[::-1]
    out = np.full(rho.shape[0], C.DEPTH)
    for n in range(rho.shape[0]):
        x = rho[n] - rho[n, 0] - drho
        k = np.nonzero(x > 0)[0]
        if k.size:
            k = k[0]
            out[n] = d[k - 1] + (d[k] - d[k - 1]) * (-x[k - 1]) / (x[k] - x[k - 1])
    return out


def _finish(r: dict) -> dict:
    r["sst"], r["sss"] = r["T"][:, -1], r["S"][:, -1]
    r["mld"] = mld(r["T"], r["S"])
    r["q_net"] = r["q_sw"] + r["q_lw"] + r["q_sens"] + r["q_lat"]
    r["tau"] = np.hypot(r["tau_x"], r["tau_y"])
    return r


# ------------------------------------------------------------------------------------------------ readers
def _mds_series(run: Path, prefix: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(iterations, field names, data (nt, nflds, ...)) of an MITgcm diagnostics stream, one file per record (the
    tiles' meta files differ in timeStepDate, so MITgcmutils.rdmds refuses to stack them)."""
    metas = sorted(glob.glob(str(run / f"{prefix}.*.001.001.meta")))
    its, data, names = [], [], []
    for m in metas:
        txt = Path(m).read_text()
        its.append(int(re.search(r"timeStepNumber = \[\s*(\d+)", txt).group(1)))
        names = re.findall(r"'(\w+)\s*'", txt.split("fldList")[1])
        nf = len(names)
        data.append(np.fromfile(m.replace(".meta", ".data"), ">f8").reshape(nf, -1))
    return np.array(its), names, np.array(data)


def read_mitgcm(days: int = C.DAYS) -> dict:
    run = C.RUNS / f"mitgcm_{days}d"
    it, names, f = _mds_series(run, "fluxDiag")
    g = {n: f[:, i, 0] for i, n in enumerate(names)}
    r = dict(t_mean=it * C.DT / 3600.0, q_sens=g["EXFhs"], q_lat=g["EXFhl"], q_lw=-g["EXFlwnet"],
             q_sw=-g["EXFswnet"], tau_x=g["EXFtaux"], tau_y=g["EXFtauy"], E=1000.0 * g["EXFevap"],
             P=1000.0 * g["EXFpreci"], hbl_mean=g["KPPhbl"])
    it, names, p = _mds_series(run, "profDiag")
    q = {n: p[:, i, :] for i, n in enumerate(names)}                  # (nt, Nr) top first
    r.update(t_snap=it * C.DT / 3600.0, T=q["THETA"][:, ::-1], S=q["SALT"][:, ::-1],
             u_s=q["UVEL"][:, 0], v_s=q["VVEL"][:, 0])
    r["hbl"] = np.concatenate([[np.nan], r["hbl_mean"]])
    return _finish(r)


def read_croco(days: int = C.DAYS) -> dict:
    import netCDF4
    run = C.RUNS / f"croco_{days}d"
    i = CROCO_I
    with netCDF4.Dataset(run / "scm_his.nc") as h, netCDF4.Dataset(run / "scm_avg.nc") as a:
        r = dict(t_snap=np.asarray(h["scrum_time"][:]) / 3600.0, T=np.asarray(h["temp"][:, :, i, i]),
                 S=np.asarray(h["salt"][:, :, i, i]),
                 u_s=0.5 * np.asarray(h["u"][:, -1, i, i - 1] + h["u"][:, -1, i, i]),
                 v_s=0.5 * np.asarray(h["v"][:, -1, i - 1, i] + h["v"][:, -1, i, i]),
                 hbl=np.asarray(h["hbl"][:, i, i]))
        tm = np.asarray(a["scrum_time"][:]) / 3600.0
        r.update(t_mean=np.round(tm + 0.5), q_sw=np.asarray(a["radsw"][:, i, i]),
                 q_lw=np.asarray(a["shflx_rlw"][:, i, i]), q_sens=np.asarray(a["shflx_sen"][:, i, i]),
                 q_lat=np.asarray(a["shflx_lat"][:, i, i]),
                 tau_x=0.5 * np.asarray(a["sustr"][:, i, i - 1] + a["sustr"][:, i, i]),
                 tau_y=0.5 * np.asarray(a["svstr"][:, i - 1, i] + a["svstr"][:, i, i]),
                 hbl_mean=np.asarray(a["hbl"][:, i, i]))
        swf = np.asarray(a["swflux"][:, i, i])                        # E − P [cm d⁻¹], volume at ρ₀ (E) / as read (P)
    s = np.load(C.RUNS / f"series_{days}d.npz")
    P_vol = 0.5 * (s["precip"][:-1] + s["precip"][1:])[: swf.size]    # hourly mean of the linear interpolation
    E_vol = swf / (100.0 * 86400.0) + P_vol
    r["E"] = C.RHO0 * E_vol                                             # CROCO: E = −Q_L/(ρ₀ L_v) (croco_bulk.md §2.9)
    r["P"] = C.RHO_FW * P_vol
    return _finish(r)


def read_pyocean(tag: str, days: int = C.DAYS) -> dict:
    o = np.load(C.RUNS / f"pyocean_{days}d" / f"{tag}.npz")
    r = dict(t_snap=o["time_snap"] / 3600.0, T=o["T"], S=o["S"], u_s=o["u"][:, -1], v_s=o["v"][:, -1],
             hbl=o["hbl"], t_mean=o["time_mean"] / 3600.0, q_sw=o["q_sw"], q_lw=o["q_lw"], q_sens=o["q_sens"],
             q_lat=o["q_lat"], tau_x=o["tau_x"], tau_y=o["tau_y"], E=C.RHO0 * o["evap"], P=C.RHO0 * o["precip"],
             wall=float(o["wall"]), spread=float(o["spread"]))
    r["hbl_mean"] = 0.5 * (r["hbl"][:-1] + r["hbl"][1:])
    return _finish(r)


def load_all(days: int = C.DAYS) -> dict[str, dict]:
    out = {}
    for k, fn in (("mitgcm", lambda: read_mitgcm(days)), ("croco", lambda: read_croco(days))):
        try:
            out[k] = fn()
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"{k}: not read ({e})")
    for tag in C.RUNS_PYOCEAN:
        p = C.RUNS / f"pyocean_{days}d" / f"{tag}.npz"
        if p.exists():
            out[tag] = read_pyocean(tag, days)
    return out


# ------------------------------------------------------------------------------------------------ metrics
def means(r: dict) -> dict[str, float]:
    m = {k: float(np.mean(r[k])) for k in FLUX_KEYS}
    m["E"] *= MM_DAY
    m["P"] *= MM_DAY
    return m


def sst_metrics(r: dict, ref: dict) -> dict[str, float]:
    n = min(r["sst"].size, ref["sst"].size)
    d = r["sst"][:n] - ref["sst"][:n]
    return dict(final=float(d[n - 1]), last_day=float(d[n - 25:n].mean()), rms=float(np.sqrt(np.mean(d * d))),
                max=float(d[np.argmax(np.abs(d))]))


def offline(scheme: str, r: dict, s: dict, relative_wind: bool = True) -> dict[str, np.ndarray]:
    """
    Hourly fluxes of a pyOCEAN bulk scheme evaluated offline on the state of run ``r`` (SST and top-layer current
    at mid-hour = mean of the bounding snapshots) and the ERA5 series ``s`` at mid-hour; the same calls as
    ``core.components.surface_fluxes.SurfaceFluxes.forcing``.
    """
    from core.components.surface_fluxes import EVAP_DENSITY
    from forcing.bulk import T0_K, bulk_ncar, bulk_ncar_fms, radiative_fluxes
    from forcing.bulk_coare import PSURF, RHO0_CROCO, bulk_coare_croco
    nt = r["t_mean"].size
    def mid(a):                      # value at mid-hour; MITgcm has no snapshot at the very end: hold the last
        a = np.asarray(a)
        if a.size < nt + 1:
            a = np.append(a, a[-1])
        return 0.5 * (a[:-1] + a[1:])[:nt]
    sst, us, vs = mid(r["sst"]), mid(r["u_s"]), mid(r["v_s"])
    if not relative_wind:
        us = vs = np.zeros_like(sst)
    a = {k: mid(s[k]) for k in ("u10", "v10", "t2m", "q2m", "msl", "sw_down", "lw_down")}
    if scheme == "coare":
        f = bulk_coare_croco(a["u10"] - us, a["v10"] - vs, a["t2m"] - T0_K, a["q2m"], sst, a["lw_down"],
                             p_sfc=PSURF, z_u=10.0, z_t=2.0, rho0=RHO0_CROCO)
        q_sw, q_lw = (1.0 - C.ALBEDO) * a["sw_down"], f.q_lw
    else:
        if scheme == "ncar":
            f = bulk_ncar_fms(a["u10"], a["v10"], a["t2m"], a["q2m"], sst + T0_K, a["msl"], u_ocean=us, v_ocean=vs,
                              z_u=10.0, z_t=2.0, z_q=2.0)
        else:
            f = bulk_ncar(a["u10"], a["v10"], a["t2m"], a["q2m"], sst, p=a["msl"], u_ocean=us, v_ocean=vs,
                          z_u=10.0, z_t=2.0)
        q_sw, q_lw = radiative_fluxes(a["sw_down"], a["lw_down"], sst, albedo=C.ALBEDO, emissivity=C.EMISSIVITY)
    out = dict(q_sw=q_sw, q_lw=q_lw, q_sens=f.q_sens, q_lat=f.q_lat, tau_x=f.tau_x, tau_y=f.tau_y,
               E=f.evap * EVAP_DENSITY[scheme], P=C.RHO_FW * mid(s["precip"]))
    out["q_net"] = out["q_sw"] + out["q_lw"] + out["q_sens"] + out["q_lat"]
    out["tau"] = np.hypot(out["tau_x"], out["tau_y"])
    return out


SCHEME = {"ly04": ("ncar_ly04", True), "fms": ("ncar", True), "coare": ("coare", True),
          "coare_croco": ("coare", False)}


def split(tag: str, runs: dict, s: dict) -> dict[str, dict[str, float]]:
    """Per flux: total (py − ref), formula (offline on ref state − ref), state (offline py − offline ref),
    approximation (offline on py state − py online); all as means over the run in the table units."""
    ref = runs[REFERENCE[tag]]
    py = runs[tag]
    sch, rel = SCHEME[tag]
    off_ref, off_py = offline(sch, ref, s, rel), offline(sch, py, s, rel)
    mr, mp = means(ref), means(py)
    scale = lambda k: MM_DAY if k in ("E", "P") else 1.0                         # noqa: E731
    out = {}
    for k in FLUX_KEYS:
        o_r, o_p = float(np.mean(off_ref[k])) * scale(k), float(np.mean(off_py[k])) * scale(k)
        out[k] = dict(total=mp[k] - mr[k], formula=o_r - mr[k], state=o_p - o_r, approx=o_p - mp[k])
    return out


def table(runs: dict) -> str:
    lines = []
    m = {k: means(v) for k, v in runs.items()}
    hdr = "| flux | " + " | ".join(NAMES[k] for k in ORDER if k in runs) + " |"
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for f in FLUX_KEYS:
        cells = []
        for k in ORDER:
            if k not in runs:
                continue
            v = m[k][f]
            txt = f"{v:.4f}" if f.startswith("tau") else f"{v:.3f}" if f in ("E", "P") else f"{v:.2f}"
            if k in REFERENCE and REFERENCE[k] in m:
                ref = m[REFERENCE[k]][f]
                txt += f" ({100 * (v - ref) / abs(ref):+.1f} %)" if ref else ""
            cells.append(txt)
        lines.append(f"| {LABEL[f]} [{UNIT[f]}] | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def state_table(runs: dict) -> str:
    lines = ["| run | SST start | SST end | ΔSST end | ΔSST last day | RMS ΔSST | max ΔSST | MLD mean | "
             "ΔMLD RMS | hbl mean | SSS end |", "|" + "---|" * 11]
    for k in ORDER + DIAG:
        if k not in runs:
            continue
        r = runs[k]
        ref = runs.get(REFERENCE.get(k, ""), None)
        sm = sst_metrics(r, ref) if ref is not None else None
        n = min(r["mld"].size, ref["mld"].size) if ref is not None else 0
        dm = (f"{np.sqrt(np.mean((r['mld'][:n] - ref['mld'][:n]) ** 2)):.2f}" if ref is not None else "—")
        lines.append(f"| {NAMES[k]} | {r['sst'][0]:.3f} | {r['sst'][-1]:.3f} | "
                     + (f"{sm['final']:+.3f} | {sm['last_day']:+.3f} | {sm['rms']:.3f} | {sm['max']:+.3f}"
                        if sm else "— | — | — | —")
                     + f" | {r['mld'].mean():.2f} | {dm} | {np.nanmean(r['hbl_mean']):.2f} | {r['sss'][-1]:.3f} |")
    return "\n".join(lines)


def split_table(runs: dict, s: dict) -> str:
    lines = ["| run | flux | total py − ref | formula at ref state | state (SST, current) | offline approx. |",
             "|---|---|---|---|---|---|"]
    for tag in ("ly04", "coare", "coare_croco", "fms"):
        if tag not in runs or REFERENCE[tag] not in runs:
            continue
        sp = split(tag, runs, s)
        for f in ("q_lw", "q_sens", "q_lat", "q_net", "tau", "E"):
            d = sp[f]
            fmt = "{:+.4f}" if f.startswith("tau") else "{:+.3f}"
            lines.append(f"| {tag} | {LABEL[f]} | " + " | ".join(fmt.format(d[x]) for x in
                                                                  ("total", "formula", "state", "approx")) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ figures
COLORS = {"mitgcm": "#2a78d6", "ly04": "#eb6834", "croco": "#1baf7a", "coare": "#eda100", "coare_croco": "#e87ba4",
          "fms": "#008300"}
STYLE = {"mitgcm": "-", "croco": "-", "ly04": "--", "coare": "--", "coare_croco": ":", "fms": "-."}


def figures(runs: dict, days: int, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e4e3df", "grid.linewidth": 0.6, "font.size": 9})
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    t0 = np.datetime64(C.START, "s")

    def dates(h):
        return t0 + (np.asarray(h) * 3600).astype("timedelta64[s]")

    for key, ylabel, fname, title in (("sst", "SST, top 1 m layer [°C]", "sst.png", "SST"),
                                      ("mld", "mixed-layer depth [m] (Δρ = 0.03 kg m⁻³)", "mld.png", "MLD")):
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=(2, 1))
        for k in ORDER:
            if k in runs:
                y = runs[k][key]
                if key == "mld":                                         # daily running mean for readability
                    y = np.convolve(y, np.ones(25) / 25, mode="same")
                axes[0].plot(dates(runs[k]["t_snap"]), y, STYLE[k], color=COLORS[k], lw=1.4, label=NAMES[k])
        for k in ("ly04", "fms", "coare", "coare_croco"):
            if k in runs and REFERENCE[k] in runs:
                ref = runs[REFERENCE[k]]
                n = min(ref[key].size, runs[k][key].size)
                d = runs[k][key][:n] - ref[key][:n]
                if key == "mld":
                    d = np.convolve(d, np.ones(25) / 25, mode="same")
                axes[1].plot(dates(runs[k]["t_snap"][:n]), d, STYLE[k], color=COLORS[k], lw=1.2,
                             label=f"{NAMES[k]} − {NAMES[REFERENCE[k]].split()[0]}")
        if key == "sst":
            axes[1].axhspan(-0.2, 0.2, color="#cde2fb", alpha=0.6, lw=0, label="criterion ±0.2 °C")
        else:
            axes[0].invert_yaxis()
        axes[0].set_ylabel(ylabel)
        axes[1].set_ylabel("pyOCEAN − reference")
        axes[0].set_title(f"{title}, ERA5 column at {C.LAT:.2f}°N {-C.LON:.2f}°W, {days} days from 2017-07-01"
                          + (" (daily running mean)" if key == "mld" else ""), loc="left")
        axes[0].legend(ncol=2, fontsize=8, frameon=False)
        axes[1].legend(ncol=2, fontsize=7, frameon=False)
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    # separation: SST difference from the reference of the bulk runs and of the prescribed-flux runs
    sep = [k for k in ("ly04", "flux_mitgcm", "coare", "flux_croco", "flux_croco_bg", "coare_bg") if k in runs]
    if any(k in runs for k in DIAG):
        sc = dict(zip(("ly04", "flux_mitgcm", "coare", "flux_croco", "flux_croco_bg", "coare_bg"),
                      ("#eb6834", "#2a78d6", "#eda100", "#1baf7a", "#6250d6", "#e34948")))
        ls = dict(ly04="--", flux_mitgcm="-", coare="--", flux_croco="-", flux_croco_bg=":", coare_bg="-.")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for ax, ref in zip(axes, ("mitgcm", "croco")):
            for k in sep:
                if REFERENCE[k] != ref:
                    continue
                n = min(runs[ref]["sst"].size, runs[k]["sst"].size)
                ax.plot(dates(runs[k]["t_snap"][:n]), runs[k]["sst"][:n] - runs[ref]["sst"][:n], ls[k], color=sc[k],
                        lw=1.3, label=NAMES[k])
            ax.axhspan(-0.2, 0.2, color="#cde2fb", alpha=0.6, lw=0, label="criterion ±0.2 °C")
            ax.set_ylabel(f"SST − {NAMES[ref].split()[0]} [°C]")
            ax.legend(ncol=2, fontsize=7, frameon=False)
        axes[0].set_title("Bulk run against the same ocean model forced by the reference's own fluxes", loc="left")
        fig.tight_layout()
        p = out / "sst_separation.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    # mean flux bars with the ±2 % band of each reference
    m = {k: means(v) for k, v in runs.items()}
    keys = ("q_sw", "q_lw", "q_sens", "q_lat", "q_net", "tau_x", "tau_y", "E")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    present = [k for k in ORDER if k in runs]
    for ax, f in zip(axes.ravel(), keys):
        x = np.arange(len(present))
        vals = [m[k][f] for k in present]
        for ref, xs in (("mitgcm", [i for i, k in enumerate(present) if k in ("mitgcm", "ly04", "fms")]),
                        ("croco", [i for i, k in enumerate(present) if k in ("croco", "coare", "coare_croco")])):
            if ref in m and xs:
                v = m[ref][f]
                ax.fill_between([min(xs) - 0.45, max(xs) + 0.45], v - 0.02 * abs(v), v + 0.02 * abs(v),
                                color="#c3c2b7", alpha=0.55, lw=0, zorder=0)
        ax.bar(x, vals, width=0.7, color=[COLORS[k] for k in present], edgecolor="white", linewidth=2, zorder=2)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_xticks(x, [k for k in present], rotation=40, ha="right", fontsize=7)
        ax.set_title(f"{LABEL[f]} [{UNIT[f]}]", loc="left")
        lo, hi = min(vals + [0]), max(vals + [0])
        pad = 0.08 * (hi - lo or 1)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Mean surface fluxes over {days} days (+ into the ocean). Grey band: ±2 % of the reference "
                 "(MITgcm for mitgcm/ly04/fms, CROCO for croco/coare/coare_croco)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    p = out / "flux_bars.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=C.DAYS)
    ap.add_argument("--figs", action="store_true")
    a = ap.parse_args()
    runs = load_all(a.days)
    s = dict(np.load(C.RUNS / f"series_{a.days}d.npz"))
    print(f"## Mean fluxes over {a.days} days (relative difference from the reference)\n")
    print(table(runs))
    print("\n## SST and mixed layer (Δ = pyOCEAN − reference)\n")
    print(state_table(runs))
    print("\n## Split of the mean flux differences\n")
    print(split_table(runs, s))
    for k, r in runs.items():
        if "wall" in r:
            print(f"\n{k}: wall {r['wall']:.0f} s, 5x5 spread {r['spread']:.1e}")
    if a.figs:
        for p in figures(runs, a.days, C.RUNS / "figs"):
            print("figure:", p)


if __name__ == "__main__":
    main()
