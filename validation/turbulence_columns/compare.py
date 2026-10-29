"""
Compare the pyOCEAN single-column runs with CROCO and with theory (N3 task 8; spec criteria 1 and 3).

    python -m validation.turbulence_columns.compare            # tables of every case found on disk

Definitions
-----------
* Mixed-layer depth D_m (Kato–Phillips; criterion 1): depth below the surface of the shallowest interface where
  k < 1e-5 m² s⁻² (Warner et al. 2005, l. 1618 of the local copy), searched from the surface down, with log₁₀ k
  interpolated linearly between the two interfaces that bracket 1e-5 (removes the Δz = 0.5 m quantisation; the same
  function for CROCO and pyOCEAN).  For KPP, which has no k, D_m(T) is the depth of the strongest temperature jump
  (``mixed_depth_T``, the entrainment interface), and the boundary-layer depth h_bl is reported separately.
* Price law (Price 1979; Kato & Phillips 1969; as written in Warner et al. 2005, eq. 60):
  D_m(t) = 1.05 u* N₀^(−1/2) t^(1/2).
* Profile errors (criterion 1, other cases): RMS of pyOCEAN − CROCO over the column at the same hour, for T, u, v
  (layers) and k, ε, ν, κ_t (interfaces), normalised by the CROCO range of the field over the column at that hour.
* σ vs ALE (criterion 3): max |ALE − σ| relative to max |σ| over the run.

CROCO's scm_his.nc: k = ``tke``, ε = ``gls`` (k-ε: ψ = c_μ⁰³ k^{3/2} l⁻¹ = ε), ν = ``AKv``, κ_t = ``AKt``, h_bl =
``hbl``; the central column (eta_rho = xi_rho = 3).
"""
from __future__ import annotations

import warnings

import numpy as np

from validation.turbulence_columns.cases import CASES, CROCO_RUNS, RUNS

PY_OUT = RUNS / "pyocean"
K_MIXED = 1e-5          # [m² s⁻²] k threshold of D_m (task 8 brief)


def price_depth(t: np.ndarray, ustar: float = 0.01, n0: float = 0.01) -> np.ndarray:
    """D_m = 1.05 u* N₀^(−1/2) t^(1/2) [m] (Price 1979; Warner et al. 2005, eq. 60)."""
    return 1.05 * ustar * np.sqrt(np.asarray(t, float) / n0)


def mixed_depth(zw: np.ndarray, k: np.ndarray, kmin: float = K_MIXED) -> float:
    """
    D_m [m] of one profile: zw (N+1,) interface heights (bed first, z ≤ 0, surface at zw[-1]), k (N+1,) [m² s⁻²].
    From the surface down, the first interface j with k_j < kmin; log₁₀ k interpolated between j and j+1.
    """
    zw, k = np.asarray(zw, float), np.maximum(np.asarray(k, float), 1e-30)
    n = len(k) - 1
    j = n
    while j > 0 and k[j - 1] >= kmin:
        j -= 1
    if j == 0:
        return float(zw[-1] - zw[0])
    # k[j] ≥ kmin > k[j-1] (j = n: the surface interface itself; k there is never below kmin when forced)
    if k[j] < kmin:
        return 0.0
    a = (np.log10(kmin) - np.log10(k[j - 1])) / (np.log10(k[j]) - np.log10(k[j - 1]))
    z = zw[j - 1] + a * (zw[j] - zw[j - 1])
    return float(zw[-1] - z)


def mixed_depth_T(zw: np.ndarray, T: np.ndarray, T_init: np.ndarray | None = None) -> float:
    """Mixed-layer depth from T [m] (for KPP, which has no k): depth of the strongest temperature jump below the
    surface, the interface between layer centres with the largest ∂T/∂z (layers thinner than 1 mm — vanished hybrid
    layers — skipped; the bottom 10 % of the column skipped, where CROCO's bottom heat flux cools the bed cell and
    makes a stronger gradient than a weak entrainment jump).  ``T_init`` is not used (kept for the call signature)."""
    zw, T = np.asarray(zw, float), np.asarray(T, float)
    h = np.diff(zw)
    zc = 0.5 * (zw[1:] + zw[:-1])
    ok = (h > 1e-3) & (zc > zw[0] + 0.1 * (zw[-1] - zw[0]))
    zc, Tk = zc[ok], T[ok]
    if len(zc) < 2:
        return 0.0
    grad = np.diff(Tk) / np.diff(zc)
    j = int(np.argmax(grad))
    return float(zw[-1] - 0.5 * (zc[j] + zc[j + 1]))


def load_croco(name: str, closure: str = "gls") -> dict | None:
    """CROCO hourly snapshots of the central column, in the layout of run_pyocean (bed first)."""
    path = CROCO_RUNS / f"{name}_{closure}" / "scm_his.nc"
    if not path.exists():
        return None
    import netCDF4
    case = CASES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = netCDF4.Dataset(path)
        j = i = 3
        out = dict(time=np.asarray(d["scrum_time"][:], float),
                   T=np.asarray(d["temp"][:, :, j, i], float),
                   u=np.asarray(d["u"][:, :, j, i], float), v=np.asarray(d["v"][:, :, j, i], float))
        for mine, theirs in (("tke", "tke"), ("eps", "gls"), ("akv", "AKv"), ("akt", "AKt")):
            out[mine] = (np.asarray(d[theirs][:, :, j, i], float) if theirs in d.variables
                         else np.full((len(out["time"]), case.n + 1), np.nan))
        out["hbl"] = np.asarray(d["hbl"][:, j, i], float) if "hbl" in d.variables else np.full(len(out["time"]), np.nan)
        d.close()
    zw = -case.depth + np.arange(case.n + 1) * case.dz
    out["zw"] = np.broadcast_to(zw, (len(out["time"]), case.n + 1)).copy()
    return out


def load_py(name: str, coord: str, closure: str = "gls", production: str = "croco") -> dict | None:
    tag = f"{name}_{coord}_{closure}" + ("_energy" if production == "energy" else "")
    path = PY_OUT / f"{tag}.npz"
    if not path.exists():
        return None
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def dm_series(run: dict) -> np.ndarray:
    return np.array([mixed_depth(z, k) for z, k in zip(run["zw"], run["tke"])])


def dm_T_series(run: dict) -> np.ndarray:
    return np.array([mixed_depth_T(z, T, run["T"][0]) for z, T in zip(run["zw"], run["T"])])


def _on_croco_levels(run: dict, field: str, it: int, ref_zw: np.ndarray) -> np.ndarray:
    """A pyOCEAN profile at hour ``it`` on CROCO's uniform levels (layers: centres; interfaces: zw); the ALE
    hybrid grid moves, so its profile is interpolated, σ/z* are on the same levels already."""
    z = run["zw"][it]
    val = run[field][it]
    if val.shape[0] == z.shape[0]:                                   # interface field
        zs, zt = z, ref_zw
    else:
        zs, zt = 0.5 * (z[1:] + z[:-1]), 0.5 * (ref_zw[1:] + ref_zw[:-1])
    ok = np.isfinite(val)
    if np.allclose(zs, zt):
        return val
    zs_u, idx = np.unique(zs[ok], return_index=True)
    return np.interp(zt, zs_u, val[ok][idx])


def profile_errors(py: dict, cr: dict, hours: list[int], fields=("T", "u", "v", "tke", "eps", "akv", "akt")) -> dict:
    """{field: [(hour, rms/range, rms), …]} of pyOCEAN − CROCO at the given hours."""
    res = {}
    for f in fields:
        rows = []
        for h in hours:
            it_p = int(np.argmin(np.abs(py["time"] - 3600.0 * h)))
            it_c = int(np.argmin(np.abs(cr["time"] - 3600.0 * h)))
            ref = cr[f][it_c]
            if not np.isfinite(ref).all() or not np.isfinite(py[f][it_p]).all():
                continue
            mine = _on_croco_levels(py, f, it_p, cr["zw"][it_c])
            sl = slice(1, -1) if mine.shape[0] == cr["zw"].shape[1] else slice(None)   # interfaces: skip bed/surf
            if f in ("u", "v"):
                sl = slice(None)
            d = mine[sl] - ref[sl]
            rms = float(np.sqrt(np.mean(d**2)))
            rng = float(np.ptp(ref[sl])) or float(np.abs(ref[sl]).max()) or 1.0
            rows.append((h, rms / rng, rms))
        res[f] = rows
    return res


def sigma_vs_ale(a: dict, s: dict, fields=("T", "u", "v", "tke", "eps", "akv", "akt")) -> dict:
    """max |ALE − σ| / max |σ| over the run, per field (same levels: z* with fixed layers)."""
    out = {}
    for f in fields:
        x, y = a[f], s[f]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        out[f] = float(np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-300))
    return out


def kato_table() -> list[str]:
    lines = ["| run | D_m 6 h | 12 h | 24 h | 30 h | vs Price (30 h) | vs CROCO (30 h) |",
             "|---|---|---|---|---|---|---|"]
    cr = load_croco("kato_phillips")
    hours = [6, 12, 24, 30]
    price = price_depth(np.array(hours) * 3600.0)
    lines.append("| Price 1979 | " + " | ".join(f"{p:.2f}" for p in price) + " | — | — |")
    dmc = None
    if cr is not None:
        dmc = dm_series(cr)
        vals = [dmc[int(np.argmin(np.abs(cr['time'] - 3600 * h)))] for h in hours]
        lines.append("| CROCO GLS | " + " | ".join(f"{v:.2f}" for v in vals)
                     + f" | {100 * (vals[-1] / price[-1] - 1):+.1f} % | — |")
    for coord in ("sigma", "zstar", "hybrid"):
        for prod in ("croco", "energy"):
            py = load_py("kato_phillips", coord, "gls", prod)
            if py is None:
                continue
            dm = dm_series(py)
            vals = [dm[int(np.argmin(np.abs(py['time'] - 3600 * h)))] for h in hours]
            vc = f"{100 * (vals[-1] / dmc[-1] - 1):+.1f} %" if dmc is not None else "—"
            lines.append(f"| pyOCEAN {coord} GLS P={prod} | " + " | ".join(f"{v:.2f}" for v in vals)
                         + f" | {100 * (vals[-1] / price[-1] - 1):+.1f} % | {vc} |")
    return lines


# ---------------------------------------------------------------- per-case scalar diagnostics
def _at(run: dict, hour: float) -> int:
    return int(np.argmin(np.abs(run["time"] - 3600.0 * hour)))


def depth_mean(run: dict, field: str, it: int) -> float:
    h = np.diff(run["zw"][it])
    return float((h * run[field][it]).sum() / h.sum())


def case_diagnostics(name: str, run: dict) -> dict:
    """Scalar diagnostics of one run (CROCO or pyOCEAN) at the end of the case (see the README table)."""
    end = len(run["time"]) - 1
    out = {}
    if name in ("kato_phillips", "willis_deardorff"):
        out["D_m(k) [m]"] = mixed_depth(run["zw"][end], run["tke"][end]) if np.isfinite(run["tke"][end]).all() \
            else np.nan
        out["D_m(T) [m]"] = mixed_depth_T(run["zw"][end], run["T"][end], run["T"][0])
        out["SST [°C]"] = float(run["T"][end][-1])
    elif name == "diurnal_cycle":
        last = run["time"] >= run["time"][end] - 86400.0 + 1.0
        sst = run["T"][last][:, -1]
        out["SST max−min day 8 [K]"] = float(sst.max() - sst.min())
        out["SST mean day 8 [°C]"] = float(sst.mean())
        it = _at(run, run["time"][end] / 3600.0 - 18.0)              # 06:00 of day 8, end of the night convection
        out["D_m(T) 06 h day 8 [m]"] = mixed_depth_T(run["zw"][it], run["T"][it])
        out["D_m(k) 06 h day 8 [m]"] = mixed_depth(run["zw"][it], run["tke"][it]) \
            if np.isfinite(run["tke"][it]).all() else np.nan
    elif name in ("forced_ekbbl", "forced_dbleek"):
        out["ū (x) [m/s]"] = depth_mean(run, "u", end)
        out["v̄ (y) [m/s]"] = depth_mean(run, "v", end)
        out["u surf [m/s]"] = float(run["u"][end][-1])
        out["v surf [m/s]"] = float(run["v"][end][-1])
        ub, vb = run["u"][end][0], run["v"][end][0]
        out["bed-layer angle [deg]"] = float(np.degrees(np.arctan2(vb, ub)))
        out["k max [m²/s²]"] = float(np.nanmax(run["tke"][end]))
    elif name == "forced_nonrotbbl":
        out["v̄ [m/s]"] = depth_mean(run, "v", end)
        out["v surf [m/s]"] = float(run["v"][end][-1])
        out["ν max [m²/s]"] = float(np.nanmax(run["akv"][end]))
    elif name == "forced_oscnonrotbbl":
        T = 12.4 * 3600.0
        last = run["time"] >= run["time"][end] - T + 1.0
        tt = run["time"][last]
        vbar = np.array([depth_mean(run, "v", i) for i in np.where(last)[0]])
        A = np.linalg.lstsq(np.c_[np.cos(2 * np.pi * tt / T), np.sin(2 * np.pi * tt / T), np.ones_like(tt)],
                            vbar, rcond=None)[0]
        out["v̄ amplitude [m/s]"] = float(np.hypot(A[0], A[1]))
        out["v̄ phase [deg]"] = float(np.degrees(np.arctan2(A[1], A[0])))
        out["ν max last cycle [m²/s]"] = float(np.nanmax(run["akv"][last]))
    return out


def gls_case_table(name: str) -> list[str]:
    cr = load_croco(name)
    rows = [("CROCO", cr)] + [(f"pyOCEAN {c}", load_py(name, c)) for c in ("sigma", "zstar", "hybrid")]
    rows = [(lab, r) for lab, r in rows if r is not None]
    if not rows:
        return []
    diag = [(lab, case_diagnostics(name, r)) for lab, r in rows]
    keys = list(diag[0][1])
    lines = [f"**{name}**", "", "| run | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for lab, d in diag:
        lines.append(f"| {lab} | " + " | ".join(f"{d[k]:.4g}" for k in keys) + " |")
    if cr is not None:
        hours = sorted({int(round(CASES[name].hours / 2)), int(round(CASES[name].hours))})
        lines += ["", f"RMS(pyOCEAN − CROCO)/range(CROCO) at hours {hours}:", "",
                  "| run | T | u | v | k | ε | ν | κ_t |", "|---|---|---|---|---|---|---|---|"]
        for lab, r in rows[1:]:
            err = profile_errors(r, cr, hours)
            cells = []
            for f in ("T", "u", "v", "tke", "eps", "akv", "akt"):
                vals = [e[1] for e in err.get(f, [])]
                cells.append("/".join(f"{v:.3f}" for v in vals) if vals else "—")
            lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    return lines + [""]


def sigma_ale_table() -> list[str]:
    lines = ["| case | coord | T | u | v | k | ε | ν | κ_t |", "|---|---|---|---|---|---|---|---|---|"]
    for name in CASES:
        s = load_py(name, "sigma")
        if s is None:
            continue
        for coord in ("zstar", "hybrid"):
            a = load_py(name, coord)
            if a is None:
                continue
            d = sigma_vs_ale(a, s) if coord == "zstar" else {
                f: float(np.max(np.abs(np.array([_on_croco_levels(a, f, i, s["zw"][i]) for i in range(len(s["time"]))])
                                       - s[f])) / max(np.max(np.abs(s[f])), 1e-300))
                for f in ("T", "u", "v", "tke", "eps", "akv", "akt")}
            lines.append(f"| {name} | {coord} | " + " | ".join(f"{d.get(f, np.nan):.1e}" for f in
                                                               ("T", "u", "v", "tke", "eps", "akv", "akt")) + " |")
    return lines


def kpp_table() -> list[str]:
    lines = ["| case | run | h_bl end [m] | D_m(T) end [m] | SST end [°C] | Price D_m [m] |",
             "|---|---|---|---|---|---|"]
    for name in ("kato_phillips", "willis_deardorff", "diurnal_cycle"):
        runs = [("CROCO-KPP", load_croco(name, "kpp")), ("CROCO-GLS", load_croco(name, "gls"))] + \
            [(f"pyOCEAN {co} KPP", load_py(name, co, "kpp")) for co in ("sigma", "zstar")]
        for lab, r in runs:
            if r is None:
                continue
            end = len(r["time"]) - 1
            price = f"{price_depth(r['time'][end]):.2f}" if name == "kato_phillips" else "—"
            hbl = "—" if "GLS" in lab else f"{r['hbl'][end]:.2f}"
            lines.append(f"| {name} | {lab} | {hbl} | "
                         f"{mixed_depth_T(r['zw'][end], r['T'][end], r['T'][0]):.2f} | {r['T'][end][-1]:.4f} | "
                         f"{price} |")
    return lines


def wall_table() -> list[str]:
    lines = ["| run | steps | wall [s] | ms/step |", "|---|---|---|---|"]
    for p in sorted(PY_OUT.glob("*.npz")):
        with np.load(p) as f:
            lines.append(f"| {p.stem} | {int(f['n_steps'])} | {float(f['wall']):.1f} | "
                         f"{1e3 * float(f['s_per_step']):.1f} |")
    return lines


def main() -> None:
    print("## Kato–Phillips D_m [m] (k < 1e-5)\n")
    print("\n".join(kato_table()))
    print("\n## GLS cases vs CROCO\n")
    for name in CASES:
        print("\n".join(gls_case_table(name)))
    print("\n## σ vs ALE: max |ALE − σ| / max |σ| over the run\n")
    print("\n".join(sigma_ale_table()))
    print("\n## KPP\n")
    print("\n".join(kpp_table()))
    print("\n## Wall times (5 × 5 periodic column, Δt = 30 s)\n")
    print("\n".join(wall_table()))


if __name__ == "__main__":
    main()
