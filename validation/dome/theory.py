"""
Theoretical diagnostic of the DOME plume (BBL drag-mixing plan, Task 6; spec
docs/superpowers/specs/2026-10-30-bbl-drag-mixing-design.md goal A, criterion 5; reading notes
docs/design/bbl_refs/entrainment_theory.md).  Post-processing only, on the common daily series
(validation/dome/series.py: layers surface first, h ≥ 0, land NaN, u on the nx+1 x-faces, v on the ny+1 y-faces),
the same for every model.

The plume flows west along the slope (x decreasing; along-slope speed −u) and descends to the south (down-slope
speed −v); x' = x − 1050 km, y' = y − 600 km as Legg et al. (2006).  Library files are relative to
/Users/estebancruz/Documents/TESIS_InternalWaves-2/Literatura/; [inferencia] marks what is inferred, not read.

Operational definitions (chosen here; the reasoning is in the docstrings and in the Task-6 report):

* **Column on the path.**  For every record and every x, the row j nearest the dye centroid Y_n(x) of that record
  (Legg eq. 18, ``metrics.path_Y`` on the single record): the quantities follow the plume as it meanders, instead of a
  fixed row of the time-mean path that the plume may have left.
* **Plume.**  The bed-attached run of layers with s > thr (thr = 0.01, Legg eq. 11), walking up from the bed and
  stopping at the first layer with s ≤ thr.  Layers thinner than ``h_min`` (0.1 m; the hybrid's vanished layers are
  1 cm) are transparent: they neither extend nor stop the plume and carry no weight in the means (their T and u are not
  physical), but their thickness counts in the height.  h is the height of the plume top above the bed.
* **Ambient "just above".**  The thickness-weighted mean over the water in the band [h, h + d_amb] above the bed
  (d_amb = 100 m by default), overlap-weighted so it is the same for z, σ and isopycnal layers.  100 m is a third of
  the inflow height h₀ = 300 m and of the order of the 10-km plume thickness, and it averages over at most one 144-m z
  level, so Δb is the jump at the interface and not the ambient stratification N² over a plume height.
* **Bottom Ekman transport.**  Ezer (2005; Ezer_2004*.md §3, "about 2 Sv, or 22 %") and Petersen et al. (2015b
  §5, Fig. 15) define it as the difference of the down-slope transport between a run and one with ν_v = 0, which needs a
  second run.  Here it is a single-run measure: the down-slope transport in the bottom ``d`` metres (the drag layer
  hbbl = 10 m of the case by default) in excess of the plume interior above it,
      Q_E(x) = Σ_y ∫_0^d −(v − v_int) dz Δy,   v_int = mean v of the attached plume in [d, h],
  over the columns whose bed layer holds plume water.  It is zero for a plume without vertical shear (no Ekman veering)
  and counts only the frictional, ageostrophic part.  Columns with h ≤ d have no interior and add only to the raw
  bottom transport Q_bbl = Σ_y ∫_0^d −v dz Δy.  A z level of 144 m cannot resolve a 10-m layer: there Q_E ≈ 0 by
  construction (the bed cell holds one velocity), so Q_E is a lower bound in coarse z/z* runs [inferencia].
"""
from __future__ import annotations

import warnings

import numpy as np

from validation.dome import cases, metrics

RI_C = 0.25                    # critical gradient Richardson number (Xu 2006 eq. 16; Jackson et al. 2008 §6)
F0 = cases.get(10.0).f0        # f-plane Coriolis of every DOME resolution [s⁻¹]
H_MIN = 0.1                    # layers thinner than this [m] are transparent (hybrid vanished layers: 1 cm)
D_AMB = 100.0                  # thickness of the ambient band above the plume top [m]
NEAR, FAR = (-180e3, -50e3), (-500e3, -200e3)   # x' windows of the table: Legg's α_E window (Fig. 3b) and downstream


def _centres(s: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """u, v of record n at the cell centres (face means), NaN → 0 [m s⁻¹], (nz, ny, nx)."""
    u, v = np.nan_to_num(s["u"][n]), np.nan_to_num(s["v"][n])
    return 0.5 * (u[..., 1:] + u[..., :-1]), 0.5 * (v[:, 1:, :] + v[:, :-1, :])


def _band_weights(h_bot: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Overlap [m] of each layer (bed first, thicknesses h_bot (nz, ny, nx)) with the band lo ≤ z ≤ hi above the bed."""
    zb = np.concatenate([np.zeros((1,) + h_bot.shape[1:]), np.cumsum(h_bot, axis=0)])
    return np.clip(np.minimum(zb[1:], hi[None]) - np.maximum(zb[:-1], lo[None]), 0.0, None)


def _record_fields(s: dict, n: int, thr: float, h_min: float, d_amb: float) -> dict:
    """
    Plume and ambient fields of record n on the whole (ny, nx) grid (module docstring): plume height h [m], plume-mean
    u_p, v_p [m s⁻¹] and b_p [m s⁻²], ambient u_a, v_a, b_a over [h, h + d_amb], the along-slope velocity −u of the
    lowest non-thin layer (u_bed) and ``plume`` (bed layer holds plume water).  Arrays are bed first internally.
    """
    h = np.nan_to_num(s["h"][n])[::-1]
    sn = np.nan_to_num(s["s"][n])[::-1]
    b = metrics.buoyancy(np.nan_to_num(s["T"][n]))[::-1]
    uc, vc = (q[::-1] for q in _centres(s, n))
    act = h >= h_min
    stop = act & (sn <= thr)
    below = np.cumsum(stop, axis=0) == 0                                # layers under the first stop layer
    in_pl = below & act & (sn > thr)
    plume = in_pl.any(axis=0)
    hp = np.where(plume, (h * below).sum(axis=0), 0.0)                  # height of the plume top above the bed
    w = h * in_pl
    W = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = {k: np.where(plume, (w * q).sum(axis=0) / W, np.nan)
                  for k, q in (("u_p", uc), ("v_p", vc), ("b_p", b))}
        wa = _band_weights(h, hp, hp + d_amb) * act
        Wa = wa.sum(axis=0)
        ok = plume & (Wa > 0)
        mean_a = {k: np.where(ok, (wa * q).sum(axis=0) / Wa, np.nan) for k, q in (("u_a", uc), ("v_a", vc), ("b_a", b))}
    k_bed = np.argmax(act, axis=0)                                      # lowest non-thin layer
    u_bed = -np.take_along_axis(uc, k_bed[None], axis=0)[0]
    return dict(h=np.where(plume, hp, np.nan), plume=plume, u_bed=np.where(plume, u_bed, np.nan), **mean_p, **mean_a)


def _path_rows(s: dict, n: int) -> np.ndarray:
    """Row index nearest the dye centroid Y_n(x) of record n for every x; −1 where the column holds no dye."""
    t = s["t"][n]
    _, Y = metrics.path_Y(s, t, t)
    j = np.argmin(np.abs(s["y"][:, None] - np.nan_to_num(Y)[None, :]), axis=0)
    return np.where(np.isfinite(Y), j, -1)


def _at_path(field: np.ndarray, j: np.ndarray) -> np.ndarray:
    """field[j(x), x] with NaN where j < 0."""
    i = np.arange(j.size)
    return np.where(j >= 0, field[np.maximum(j, 0), i], np.nan)


def _nanmean(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def plume_bulk(s: dict, t0: float, t1: float, thr: float = 0.01, h_min: float = H_MIN,
               d_amb: float = D_AMB) -> dict[str, np.ndarray]:
    """
    Bulk plume properties along the path, per x (time means over the records in [t0, t1] where the column on the path
    holds an attached plume; definitions in the module docstring):

        h        plume height above the bed [m]
        dU       |U_p − U_a|, plume-mean minus ambient horizontal velocity [m s⁻¹]
        db       Δb = b_a − b_p > 0 for a dense plume [m s⁻²]
        RiB      Δb h / ΔU²   (bulk Richardson number, Xu 2006 eq. 4; Legg 2006 eq. 2)
        Fr       ΔU / √(Δb h) (Ezer 2005 eq. 3) = RiB^(−1/2)
        u_along  plume-mean along-slope speed −u_p [m s⁻¹]

    RiB and Fr are formed from the time means of h, ΔU and Δb (the bulk state, not the mean of a ratio).

    Caveat (Task-6 review I2): Ri_B depends strongly on the plume threshold: from s > 0.01 to s > 0.1 it
    falls ×1.4–4.6 in −180 < x' < −50 km (MITgcm legg 9.1 → 6.5, hybrid kv_low + Smag + Jackson 6.5 → 1.4), because
    the dilute envelope enters h and the plume means.  With ΔU against the 100 m just above, it is not Legg's bulk
    Ri_B ≈ 0.4–0.5 (Legg gives no operational ΔU) [inferencia]: compare runs with each other only, with one
    threshold.  E_ET(Ri_B) = 0 for Ri_B ≥ 0.8 then follows from this definition, not from the model.  Also x, xp
    (= x − 1050 km) [m] and y_path, the mean row position on the path [m].
    """
    acc = {k: [] for k in ("h", "dU", "db", "u_along", "y")}
    for n in metrics._window(s, t0, t1):
        f = _record_fields(s, n, thr, h_min, d_amb)
        j = _path_rows(s, n)
        dU = np.hypot(f["u_p"] - f["u_a"], f["v_p"] - f["v_a"])
        for k, q in (("h", f["h"]), ("dU", dU), ("db", f["b_a"] - f["b_p"]), ("u_along", -f["u_p"])):
            acc[k].append(_at_path(q, j))
        acc["y"].append(np.where(j >= 0, s["y"][np.maximum(j, 0)], np.nan))
    m = {k: _nanmean(np.array(v)) for k, v in acc.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        RiB = m["db"] * m["h"] / m["dU"] ** 2
        Fr = m["dU"] / np.sqrt(m["db"] * m["h"])
    return dict(x=s["x"], xp=s["x"] - metrics.X_INLET, y_path=m["y"], h=m["h"], dU=m["dU"], db=m["db"],
                RiB=RiB, Fr=Fr, u_along=m["u_along"])


def column_interface_ri(h: np.ndarray, b: np.ndarray, u: np.ndarray, v: np.ndarray, sc: np.ndarray,
                        thr: float = 0.01, h_min: float = H_MIN) -> np.ndarray:
    """
    Gradient Richardson numbers Ri = N²/S² at the interfaces of one column (layers surface first) that lie in the
    plume interface band thr < s_i < 1 − thr, s_i = ½(s_above + s_below):
        N² = (b_above − b_below)/δz,  S² = |U_above − U_below|²/δz²,  δz = ½(h_above + h_below)
    between consecutive layers thicker than ``h_min`` (thinner ones are skipped, as in the model's thin-layer merge).
    S² = 0 gives +inf; N² ≤ 0 gives Ri ≤ 0.  Returns the Ri of the band interfaces, top to bottom.
    """
    k = np.nonzero(h >= h_min)[0]
    if k.size < 2:
        return np.empty(0)
    a, c = k[:-1], k[1:]
    si = 0.5 * (sc[a] + sc[c])
    band = (si > thr) & (si < 1.0 - thr)
    a, c = a[band], c[band]
    dz = 0.5 * (h[a] + h[c])
    N2 = (b[a] - b[c]) / dz
    S2 = ((u[a] - u[c]) ** 2 + (v[a] - v[c]) ** 2) / dz ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(S2 > 0, N2 / np.where(S2 > 0, S2, 1.0), np.inf)


RI_BINS = np.concatenate([[-np.inf], np.linspace(-2.0, 2.0, 17), [np.inf]])   # log10 Ri bins; Ri ≤ 0 in the first


def interface_richardson(s: dict, t0: float, t1: float, thr: float = 0.01, h_min: float = H_MIN) -> dict:
    """
    Gradient Ri across the plume interface (``column_interface_ri``) at the column on the path, per x and record in
    [t0, t1].  Returns, per x (time means over the records with dye at x): ``n_low``, the number of interfaces with
    Ri < 0.25 across the interface (Jackson et al. 2008 §6 asks for ≥ 2–4 points in the low-Ri region; Jackson_2008*.md
    §6), ``n_band`` the band interfaces, ``ri_median`` the median band Ri (pooled over the records); and, over all
    columns and records, ``hist`` counts in the log10-Ri bins ``bins`` (RI_BINS; Ri ≤ 0 in the first), ``ri_samples``
    (every band Ri, in record, x, top-down order) and ``frac_low`` (share of band interfaces with Ri < 0.25).
    """
    nx = s["x"].size
    n_low, n_band, pooled, samples = [], [], [[] for _ in range(nx)], []
    for n in metrics._window(s, t0, t1):
        j = _path_rows(s, n)
        h = np.nan_to_num(s["h"][n])
        sn = np.nan_to_num(s["s"][n])
        b = metrics.buoyancy(np.nan_to_num(s["T"][n]))
        uc, vc = _centres(s, n)
        lo, bd = np.full(nx, np.nan), np.full(nx, np.nan)
        for i in np.nonzero(j >= 0)[0]:
            ri = column_interface_ri(h[:, j[i], i], b[:, j[i], i], uc[:, j[i], i], vc[:, j[i], i], sn[:, j[i], i],
                                     thr, h_min)
            lo[i], bd[i] = float((ri < RI_C).sum()), float(ri.size)
            pooled[i].extend(ri.tolist())
            samples.extend(ri.tolist())
        n_low.append(lo)
        n_band.append(bd)
    samples = np.array(samples, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(samples > 0, np.log10(np.where(samples > 0, samples, 1.0)), -np.inf)
    hist, _ = np.histogram(np.clip(lr, -1e300, 1e300), bins=np.clip(RI_BINS, -1e300, 1e300))
    med = np.array([np.median(p) if p else np.nan for p in pooled])
    return dict(x=s["x"], xp=s["x"] - metrics.X_INLET, n_low=_nanmean(np.array(n_low)),
                n_band=_nanmean(np.array(n_band)), ri_median=med, bins=RI_BINS, hist=hist, ri_samples=samples,
                frac_low=float((samples < RI_C).mean()) if samples.size else float("nan"))


def legg_numbers(nu_v: float | None, cd: float) -> dict:
    """
    Nondimensional numbers of Legg et al. (2006) Table 2 (Case 1) from the case parameters of cases.py
    (Δb₀ = gΔρ/ρ₀, h₀, N² = Δb₀/H, f, channel width W, slope tan α of the bathymetry):
        Fr = U₀/√(Δb₀h₀) (1),  Δb₀/(N²h₀) (12),  stretching L_ρ tan α/h₀ (0.8),  W/L_ρ (4.1),  N/f (23.0),
        drag Ekman C_d Δb₀ tan α/(f² h₀) (0.127),  frictional Ekman ν_v/(h₀² f)
    with U₀ = √(Δb₀h₀), L_ρ = U₀/f.  Frictional Ekman > 0.1: laminar flow without eddies (Cenedese et al. 2004 in
    Legg §2, p. 74).  ``nu_v`` None (a closure computes ν) gives NaN for it.
    """
    c = cases.get(10.0)
    db0 = cases.G * cases.DRHO / cases.RHO0
    U0 = np.sqrt(db0 * cases.H0)
    N2 = db0 / cases.HMAX
    Lr = U0 / c.f0
    tana = float(np.nanmax(np.abs(np.diff(c.H, axis=0))) / c.dx)       # the 1 % slope of DOME_initialize_topography
    W = float(((c.x > 1000e3) & (c.x < 1100e3)).sum() * c.dx)
    return dict(Fr=U0 / np.sqrt(db0 * cases.H0), db_ratio=db0 / (N2 * cases.H0), stretching=Lr * tana / cases.H0,
                width=W / Lr, N_over_f=np.sqrt(N2) / c.f0, drag_ekman=cd * db0 * tana / (c.f0 ** 2 * cases.H0),
                frictional_ekman=float("nan") if nu_v is None else nu_v / (cases.H0 ** 2 * c.f0), tan_alpha=tana,
                L_rho=Lr)


def bottom_ekman_transport(s: dict, t0: float, t1: float, d: float = 10.0, thr: float = 0.01,
                           h_min: float = H_MIN) -> dict:
    """
    Down-slope transport in the bottom ``d`` metres per x (module docstring; default d = hbbl = 10 m of the case),
    time mean over [t0, t1] [m³ s⁻¹], positive down-slope (−v):
        bbl     Q_bbl = Σ_y ∫_0^d −v dz Δy over the columns whose bed layer holds plume water;
        excess  Q_E = Σ_y ∫_0^d −(v − v_int) dz Δy, v_int the mean v of the attached plume in [d, h] (columns with
                h > d), the frictional (Ekman) part;
        plume   Σ_y ∫_A −v dz Δy over all plume cells (s > thr), the plume's down-slope transport;
        ratio   excess / plume where plume > 0 (Ezer 2005: ~22 % at y = 100 km, Ezer_2004*.md §3).
    """
    acc = {k: [] for k in ("bbl", "excess", "plume")}
    for n in metrics._window(s, t0, t1):
        f = _record_fields(s, n, thr, h_min, D_AMB)
        h = np.nan_to_num(s["h"][n])[::-1]
        sn = np.nan_to_num(s["s"][n])[::-1]
        _, vc = _centres(s, n)
        vc = vc[::-1]
        act = h >= h_min
        hp = np.nan_to_num(f["h"])
        zero = np.zeros_like(hp)
        wb = _band_weights(h, zero, np.minimum(np.full_like(hp, d), hp)) * act * f["plume"][None]
        wi = _band_weights(h, np.full_like(hp, d), hp) * act * (hp > d)[None]
        Wi = wi.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            v_int = np.where(Wi > 0, (wi * vc).sum(axis=0) / np.where(Wi > 0, Wi, 1.0), np.nan)
        dy = s["dy"][:, None]
        q_bbl = (wb * -vc).sum(axis=0) * dy
        q_ex = np.where(Wi > 0, (wb * -(vc - np.nan_to_num(v_int)[None])).sum(axis=0), 0.0) * dy
        q_pl = (np.where(sn > thr, -vc * h, 0.0)).sum(axis=0) * dy
        for k, q in (("bbl", q_bbl), ("excess", q_ex), ("plume", q_pl)):
            acc[k].append(q.sum(axis=0))
    m = {k: np.mean(v, axis=0) for k, v in acc.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(m["plume"] > 0, m["excess"] / np.where(m["plume"] > 0, m["plume"], 1.0), np.nan)
    return dict(x=s["x"], xp=s["x"] - metrics.X_INLET, ratio=ratio, **m)


def nof_speed(s: dict, t0: float, t1: float, thr: float = 0.01, h_min: float = H_MIN, d_amb: float = D_AMB,
              f: float = F0, min_slope: float = 1e-4) -> dict:
    """
    Along-slope plume speed against Nof's (1983) u_t = g′ s / f (Ezer 2005 eq. 2, Ezer_2004*.md p. 160), on the
    column on the path, per x, time means over [t0, t1]: g′ = Δb of ``plume_bulk`` (plume vs the ambient just above),
    s the local bed slope |∂H/∂y|.  Returns u_nof, u_plume (plume-mean −u), u_bed (−u of the lowest non-thin layer)
    [m s⁻¹] and ratio = u_plume/u_nof.  Where the bed is flatter than ``min_slope`` (abyssal plain, channel) u_nof is
    NaN.  Ezer finds the model bottom speed close to u_t (0.4 vs 0.5 and 0.23 vs 0.2 m s⁻¹).
    """
    Hn = np.where(np.isfinite(s["H"]), s["H"], np.nan)
    slope = np.abs(np.gradient(Hn, s["y"], axis=0)) if s["y"].size > 1 else np.zeros_like(Hn)
    acc = {k: [] for k in ("u_nof", "u_plume", "u_bed")}
    for n in metrics._window(s, t0, t1):
        fl = _record_fields(s, n, thr, h_min, d_amb)
        j = _path_rows(s, n)
        sl = _at_path(slope, j)
        gp = _at_path(fl["b_a"] - fl["b_p"], j)
        acc["u_nof"].append(np.where(sl > min_slope, gp * sl / f, np.nan))
        acc["u_plume"].append(_at_path(-fl["u_p"], j))
        acc["u_bed"].append(_at_path(fl["u_bed"], j))
    m = {k: _nanmean(np.array(v)) for k, v in acc.items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = m["u_plume"] / m["u_nof"]
    return dict(x=s["x"], xp=s["x"] - metrics.X_INLET, ratio=ratio, **m)


def entrainment_laws(ri) -> dict:
    """
    Entrainment laws E = w_E/ΔU against Ri (Ri < 0 taken as 0; NaN Ri, i.e. no plume, gives NaN, not a prediction):
        Ellison–Turner (Turner 1986; Xu et al. 2006 eqs. 3–4, Legg eq. 2): E = (0.08 − 0.1 Ri)/(1 + 5 Ri) for
            Ri < 0.8, else 0 — laboratory, no rotation, with the bulk Ri_B;
        Xu et al. (2006) eq. 16: E = 0.20 (1 − Ri/0.25) for Ri < 0.25, else 0 — fitted to Nek5000 (no rotation),
            used with a layer or gradient Ri.
    Legg §3.3 (p. 85): at his Ri_B ≈ 0.4–0.5 these laws give E = 8e-3–4e-2 and the rotating plume's entrainment is
    "considerably smaller"; against the corner E 8.6e-4 that is 9–50× [inferencia: the factor is ours, not Legg's].
    They explain trends, not targets (spec anti-goal).
    """
    ri = np.asarray(ri, float)
    r = np.maximum(ri, 0.0)
    et = np.where(np.isnan(ri), np.nan, np.where(r < 0.8, (0.08 - 0.1 * r) / (1.0 + 5.0 * r), 0.0))
    xu = np.where(np.isnan(ri), np.nan, np.where(r < RI_C, 0.20 * (1.0 - r / RI_C), 0.0))
    return dict(ellison_turner=et, xu=xu)


def run_nu_v(meta: dict, case: cases.Case | None = None) -> float | None:
    """Constant vertical viscosity of a run from its series meta [m² s⁻¹]; None when a closure (GLS) computes ν.
    MITgcm/MOM6: 'kv_low' 1e-4, else the case ν_v (mitgcm.py data_text, mom6.py KV)."""
    case = case or cases.get(10.0)
    if meta.get("model") == "pyOCEAN":
        if meta.get("closure") == "gls":
            return None
        return meta.get("nu_v", case.nu_v)
    return 1e-4 if meta.get("variant") == "kv_low" else case.nu_v


def _win(xp: np.ndarray, f: np.ndarray, w: tuple[float, float]) -> float:
    m = (xp > w[0]) & (xp < w[1])
    return float(_nanmean(np.asarray(f, float)[m])) if m.any() else float("nan")


def _ratio_win(xp: np.ndarray, num: np.ndarray, den: np.ndarray, w: tuple[float, float]) -> float:
    """Σ num / Σ den over the x' window (a ratio of window sums: robust where den → 0 at single x)."""
    m = (xp > w[0]) & (xp < w[1])
    d = float(np.nansum(den[m]))
    return float(np.nansum(num[m])) / d if d > 0 else float("nan")


def summary(s: dict, meta: dict, t0: float, t1: float) -> dict:
    """Window means (NEAR = Legg's α_E window −180 < x' < −50 km, FAR = −500 < x' < −200 km) of the theory
    quantities of one run, for compare.py --theory."""
    pb = plume_bulk(s, t0, t1)
    ir = interface_richardson(s, t0, t1)
    ek = bottom_ekman_transport(s, t0, t1)
    nf = nof_speed(s, t0, t1)
    x, aE = metrics.entrainment(s, t0, t1)
    xp = pb["xp"]
    out = dict(nu_v=run_nu_v(meta), frac_low=ir["frac_low"])
    for tag, w in (("near", NEAR), ("far", FAR)):
        riB, rig = _win(xp, pb["RiB"], w), _win(xp, ir["ri_median"], w)
        out[tag] = dict(h=_win(xp, pb["h"], w), dU=_win(xp, pb["dU"], w), db=_win(xp, pb["db"], w), RiB=riB,
                        Fr=_win(xp, pb["Fr"], w), n_low=_win(xp, ir["n_low"], w), n_band=_win(xp, ir["n_band"], w),
                        ri_g=rig, E_ET=float(entrainment_laws(riB)["ellison_turner"]),
                        E_Xu=float(entrainment_laws(rig)["xu"]), aE=_win(xp, aE, w),
                        ek=_ratio_win(xp, ek["excess"], ek["plume"], w), ek_sv=_win(xp, ek["excess"], w) / 1e6,
                        nof=_win(xp, nf["ratio"], w))
    return out
