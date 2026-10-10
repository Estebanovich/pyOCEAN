"""
Tide-gauge harmonic constants: loading, comparison and defect diagnosis.

Compares model (or atlas) harmonic constants with tide-gauge constants from
TICON-3 (Hart-Davis et al. 2022: least-squares analyses of the GESLA-3
records, amplitude in cm, Greenwich phase lag in degrees), and separates
instrumental defects of a gauge from genuine model–data misfit.

Defect signatures in the complex constants h = A e^{−iG}
-------------------------------------------------------
A clock offset τ (recorded time = true time + τ) gives h_obs = h e^{−iω_c τ},
a gain κ gives h_obs = κ h.  Both are fitted jointly, minimising
Σ_c |h_obs − κ h_mod e^{−iω_c τ}|², and accepted only if the correction makes
the fit good (relative misfit < 10 %) and at least halves the RSS.  The
fitted (τ, κ) are then classified:

* ``timezone``  τ within 5 min of a whole number of hours ≥ 1 — a gauge
  logged in local time (Mexico UTC−6/−7) or with a DST hour, as UTC;
* ``units``     |κ| within 5 % of 10ⁿ, n ≠ 0 — e.g. mm stored as cm;
* ``inverted``  κ < 0;
* ``lag_damping`` any other accepted (τ, κ), typically κ < 1 with a lag of
  minutes to an hour: the frequency-dependent filtering of a lagoon,
  estuary or harbour (or limited atlas resolution) — physical, not a gauge
  defect, although a slow clock cannot be excluded;
* ``misfit``    no simple correction explains a large residual.

Because physical filtering and some clock errors look alike in a single
record, ``compare_with_siblings`` also checks each record against the other
records of the same site: a record that disagrees with its siblings by a
time-zone, unit or sign signature is a gauge defect beyond reasonable doubt,
while siblings agreeing among themselves but not with the model point to the
model or to local hydrodynamics.

Metrics follow common tide-model validation practice (e.g. Stammer et al.
2014): the vector difference |h_obs − h_mod| per constituent and the
root-sum-square RSS = √(½ Σ_c |Δh_c|²) — the RMS over a tidal cycle.

References
----------
Hart-Davis, M.G., Dettmering, D. & Seitz, F. (2022). TICON-3: Tidal
    constants based on GESLA-3 sea-level records. PANGAEA,
    doi:10.1594/PANGAEA.951610.
Haigh, I.D. et al. (2023). GESLA version 3. Geosci. Data J. 10, 293–314.
Stammer, D. et al. (2014). Accuracy assessment of global barotropic ocean
    tide models. Rev. Geophys. 52, 243–282.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forcing.tides import OMEGA

_REL_MISFIT_GOOD = 0.10      # corrected misfit / signal below which a defect fit is accepted
_TIMING_MIN = 10.0           # minutes: smaller lags are not reported
_GAIN_TOL = 0.20


def complex_from_amp_phase(amp: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    """h = A e^{−iG}  (TPXO / TICON convention, G = Greenwich phase lag)."""
    return np.asarray(amp) * np.exp(-1j * np.deg2rad(phase_deg))


def vector_difference(h_obs: np.ndarray, h_mod: np.ndarray) -> np.ndarray:
    """|h_obs − h_mod| per constituent (same units as h)."""
    return np.abs(np.asarray(h_obs) - np.asarray(h_mod))


def _rss(d: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.nansum(np.abs(d) ** 2)))


def diagnose_station(h_obs: np.ndarray, h_mod: np.ndarray, constituents: list[str],
                     tau_range_h: float = 13.0) -> dict:
    """
    Compare one gauge with the model and diagnose instrumental defects.

    Parameters
    ----------
    h_obs, h_mod : np.ndarray, complex, shape (nc,)
        Observed and modelled constants [m]; NaN constituents are ignored.
    constituents : list of str
    tau_range_h : float
        Search range of the clock offset [h].

    Returns
    -------
    dict with
        rss, rss_corrected [m]; rel_misfit (corrected RSS / signal RSS);
        tau_minutes, gain (fitted τ, κ); flags (list of 'timing', 'gain',
        'inverted', 'misfit'); d_amp [m], d_phase [deg] per constituent.
    """
    h_obs = np.asarray(h_obs, complex)
    h_mod = np.asarray(h_mod, complex)
    ok = np.isfinite(h_obs) & np.isfinite(h_mod)
    w = np.array([OMEGA[c] for c in constituents])[ok]
    o, m = h_obs[ok], h_mod[ok]

    rss = _rss(o - m)
    signal = _rss(m)

    # Grid search over τ (1-min steps) with the optimal real gain κ(τ) at each τ
    taus = np.arange(-tau_range_h * 60, tau_range_h * 60 + 1) * 60.0
    rot = m[None, :] * np.exp(-1j * np.outer(taus, w))          # (ntau, nc)
    kappa = np.real(np.sum(np.conj(rot) * o[None, :], axis=1)) / np.sum(np.abs(rot) ** 2, axis=1)
    resid = np.sum(np.abs(o[None, :] - kappa[:, None] * rot) ** 2, axis=1)
    k = int(np.argmin(resid))
    tau, gain = taus[k], kappa[k]
    rss_corr = float(np.sqrt(0.5 * resid[k]))

    flags = []
    good_after = rss_corr < _REL_MISFIT_GOOD * signal and rss_corr < 0.5 * rss
    if good_after:
        hours = tau / 3600.0
        whole_hours = abs(hours) >= 1.0 - 1e-9 and abs(hours - np.round(hours)) * 60 <= 5.0
        decade = np.log10(abs(gain)) if gain != 0 else 0.0
        power_of_ten = abs(np.round(decade)) >= 1 and abs(abs(gain) / 10 ** np.round(decade) - 1) <= 0.05
        if whole_hours:
            flags.append("timezone")
        if power_of_ten:
            flags.append("units")
        if gain < 0:
            flags.append("inverted")
        if not flags and (abs(tau) / 60 > _TIMING_MIN or abs(abs(gain) - 1.0) > _GAIN_TOL):
            flags.append("lag_damping")
    if not flags and rss > _REL_MISFIT_GOOD * signal:
        flags.append("misfit")

    d_amp = np.full(len(constituents), np.nan)
    d_pha = np.full(len(constituents), np.nan)
    d_amp[ok] = np.abs(o) - np.abs(m)
    d_pha[ok] = (np.rad2deg(np.angle(m) - np.angle(o)) + 180.0) % 360.0 - 180.0   # G_obs − G_mod
    return dict(rss=rss, rss_corrected=rss_corr, rel_misfit=rss / signal if signal else np.nan,
                tau_minutes=tau / 60.0, gain=gain, flags=flags, d_amp=d_amp, d_phase=d_pha)


def compare_with_siblings(records: list[np.ndarray], constituents: list[str]) -> list[dict | None]:
    """
    Diagnose each record of one site against the median of the others.

    Uses the same classification as ``diagnose_station``, with the
    component-wise median of the sibling records (real and imaginary parts)
    in place of the model.  Returns None for a site with a single record.
    """
    if len(records) < 2:
        return [None] * len(records)
    arr = np.array(records, complex)
    out = []
    for i in range(len(arr)):
        others = np.delete(arr, i, axis=0)
        ref = np.nanmedian(others.real, axis=0) + 1j * np.nanmedian(others.imag, axis=0)
        out.append(diagnose_station(arr[i], ref, constituents))
    return out


def load_ticon(path: str | Path, lon_range: tuple[float, float], lat_range: tuple[float, float],
               constituents: list[str], types: tuple[str, ...] = ("Coastal",)) -> pd.DataFrame:
    """
    Read TICON-3 records in a region, one row per gauge record.

    Parameters
    ----------
    path : TICON_3.txt (tab separated, 14 columns; amplitude in cm,
        Greenwich phase lag in degrees, longitudes 0–360)
    lon_range, lat_range : region [°E −180–180, °N]
    constituents : lower-case names, e.g. forcing.tides.CONSTITUENTS
    types : gauge types to keep ('Coastal', 'River', 'Lake'); default
        coastal only — river and lake gauges do not see the ocean tide.

    Returns
    -------
    pandas.DataFrame with lat, lon (−180–180), start, end, miss [%], nobs,
    gap [days], src, type, hc (complex array (nc,) in m, NaN if absent) and
    sigma_amp [m], sigma_phase [deg].
    """
    cols = ["lat", "lon", "con", "amp", "pha", "samp", "spha", "miss", "nobs", "gap",
            "start", "end", "src", "type"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    df["lon"] = ((df["lon"] + 180.0) % 360.0) - 180.0
    df = df[df["lat"].between(*lat_range) & df["lon"].between(*lon_range) & df["type"].isin(types)]
    df = df.assign(con=df["con"].str.lower())

    rows = []
    keys = ["lat", "lon", "start", "end", "src", "type"]
    for key, g in df.groupby(keys, sort=False):
        g = g.set_index("con")
        hc = np.full(len(constituents), np.nan + 0j)
        s_amp = np.full(len(constituents), np.nan)
        s_pha = np.full(len(constituents), np.nan)
        for i, c in enumerate(constituents):
            if c in g.index:
                r = g.loc[[c]].iloc[0]              # TICON-3 repeats some rows
                hc[i] = complex_from_amp_phase(r["amp"] / 100.0, r["pha"])
                s_amp[i], s_pha[i] = r["samp"] / 100.0, r["spha"]
        first = g.iloc[0]
        rows.append(dict(zip(keys, key)) | dict(miss=first["miss"], nobs=first["nobs"], gap=first["gap"],
                                                hc=hc, sigma_amp=s_amp, sigma_phase=s_pha))
    out = pd.DataFrame(rows)
    if len(out):
        out["start"] = pd.to_datetime(out["start"], format="%d/%m/%Y")
        out["end"] = pd.to_datetime(out["end"], format="%d/%m/%Y")
    return out
