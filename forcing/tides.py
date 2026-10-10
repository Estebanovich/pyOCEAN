"""
Harmonic tide prediction with the OTPS / TPXO conventions.

Reconstructs elevation (or velocity / transport) time series from complex
harmonic constants, exactly as the OSU Tidal Prediction Software (OTPS) does
for the TPXO atlases (Egbert & Erofeeva 2002), so that TPXO constants — e.g.
the TPXO10-atlas-v2 values used for the HYCOM-IOA tidal ports — are applied
with their intended phase reference.

    η(t) = Σ_c f_c(t) [ Re(h_c) cos θ_c(t) − Im(h_c) sin θ_c(t) ]
    θ_c(t) = ω_c (t − t₁₉₉₂) + φ_c + u_c(t)

* h_c = Re + i·Im is the TPXO complex constant (amplitude A, Greenwich phase
  lag G: h = A e^{−iG}, so η = Σ f A cos(θ − G)).
* ω_c and φ_c (the astronomical argument at 1992-01-01 00:00) are the OTPS
  constants (``constit.h``; Egbert & Erofeeva 2002).
* f_c, u_c are the nodal corrections of OTPS, Schureman (1958)
  approximations in the lunar node N, with the mean longitudes of
  Cartwright (formulae for 1990–2010, used by OTPS at all dates).

Only the eight constituents HYCOM can force (``tidcon``) are tabulated, in
HYCOM order.  Verified against pyTMD 2.1.7 (OTIS mode) to round-off in
``tests/test_tides.py``.

References
----------
Egbert, G.D. & Erofeeva, S.Y. (2002). Efficient inverse modeling of
    barotropic ocean tides. J. Atmos. Oceanic Technol. 19, 183–204.
Schureman, P. (1958). Manual of Harmonic Analysis and Prediction of Tides.
    US Coast and Geodetic Survey, Special Publication 98.
Cartwright, D.E. (1982). Tidal analysis — a retrospect. In: Time Series
    Methods in Hydrosciences, Elsevier, 170–188.
Sutterley, T.C. et al. pyTMD: Python-based tidal prediction software
    (test oracle only).
"""

from __future__ import annotations

import numpy as np

#: Constituents in HYCOM ``tidcon`` order (digits Q1 K2 P1 N2 O1 K1 S2 M2, read right to left).
CONSTITUENTS: tuple[str, ...] = ("m2", "s2", "k1", "o1", "n2", "p1", "k2", "q1")

#: Angular frequencies [rad s⁻¹] (OTPS constit.h).
OMEGA: dict[str, float] = {
    "m2": 1.405189e-04, "s2": 1.454441e-04, "k1": 7.292117e-05, "o1": 6.759774e-05,
    "n2": 1.378797e-04, "p1": 7.252295e-05, "k2": 1.458423e-04, "q1": 6.495854e-05,
}

#: Astronomical argument at 1992-01-01 00:00 UT [rad] (OTPS constit.h).
PHASE_1992: dict[str, float] = {
    "m2": 1.731557546, "s2": 0.000000000, "k1": 0.173003674, "o1": 1.558553872,
    "n2": 6.050721243, "p1": 6.110181633, "k2": 3.487600001, "q1": 5.877717569,
}

MJD_1992: float = 48622.0           # Modified Julian Day of 1992-01-01 00:00
_MJD_UNIX: float = 40587.0          # Modified Julian Day of 1970-01-01 00:00


def to_mjd(t: np.ndarray | np.datetime64) -> np.ndarray:
    """Modified Julian Day of datetime64 values."""
    t = np.asarray(t, dtype="datetime64[ns]")
    return _MJD_UNIX + (t - np.datetime64("1970-01-01T00:00", "ns")) / np.timedelta64(1, "D")


def mean_longitudes(mjd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean longitudes [deg] of the Moon s, Sun h, lunar perigee p and lunar node N.

        T = MJD − 51544.4993                   (days from J2000, UT → TT shift)
        s = 218.3164 + 13.17639648 T
        h = 280.4661 +  0.98564736 T
        p =  83.3535 +  0.11140353 T
        N = 125.0445 −  0.05295377 T          (Cartwright; as in OTPS)
    """
    T = np.asarray(mjd, dtype=float) - 51544.4993
    s = np.mod(218.3164 + 13.17639648 * T, 360.0)
    h = np.mod(280.4661 + 0.98564736 * T, 360.0)
    p = np.mod(83.3535 + 0.11140353 * T, 360.0)
    n = np.mod(125.0445 - 0.05295377 * T, 360.0)
    return s, h, p, n


def nodal_corrections(mjd: np.ndarray, constituents: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodal amplitude factors f and phase corrections u [rad] (OTPS / Schureman 1958).

    With N the longitude of the lunar node:

        M2, N2 : f e^{iu} = 1 − 0.03731 e^{iN} + 0.00052 e^{2iN}
        K1     : f cos u = 1 + 0.1158 cos N − 0.0029 cos 2N,  f sin u = −0.1554 sin N + 0.0029 sin 2N
        K2     : f cos u = 1 + 0.2852 cos N + 0.0324 cos 2N,  f sin u = −0.3108 sin N − 0.0324 sin 2N
        O1     : f = |1 + 0.189 e^{iN} − 0.0058 e^{2iN}|,     u = 10.8° sin N − 1.3° sin 2N + 0.2° sin 3N
        Q1     : f = |1 + 0.188 e^{iN}|,                       u = arctan[0.189 sin N / (1 + 0.189 cos N)]
        S2, P1 : f = 1, u = 0

    Parameters
    ----------
    mjd : np.ndarray, shape (nt,)
        Modified Julian Days.
    constituents : list of str
        Constituent names (see ``CONSTITUENTS``).

    Returns
    -------
    f, u : np.ndarray, shape (nt, nc)
    """
    _, _, _, n = mean_longitudes(np.atleast_1d(mjd))
    N = np.deg2rad(n)
    sn, cn = np.sin(N), np.cos(N)
    s2n, c2n, s3n = np.sin(2 * N), np.cos(2 * N), np.sin(3 * N)

    f = np.empty((N.size, len(constituents)))
    u = np.empty_like(f)
    for k, c in enumerate(constituents):
        if c in ("m2", "n2"):
            t1, t2 = -0.03731 * sn + 0.00052 * s2n, 1.0 - 0.03731 * cn + 0.00052 * c2n
        elif c == "k1":
            t1, t2 = -0.1554 * sn + 0.0029 * s2n, 1.0 + 0.1158 * cn - 0.0029 * c2n
        elif c == "k2":
            t1, t2 = -0.3108 * sn - 0.0324 * s2n, 1.0 + 0.2852 * cn + 0.0324 * c2n
        elif c == "o1":
            f[:, k] = np.hypot(0.189 * sn - 0.0058 * s2n, 1.0 + 0.189 * cn - 0.0058 * c2n)
            u[:, k] = np.deg2rad(10.8 * sn - 1.3 * s2n + 0.2 * s3n)
            continue
        elif c == "q1":
            f[:, k] = np.hypot(1.0 + 0.188 * cn, 0.188 * sn)
            u[:, k] = np.arctan(0.189 * sn / (1.0 + 0.189 * cn))
            continue
        elif c in ("s2", "p1"):
            t1, t2 = np.zeros_like(N), np.ones_like(N)
        else:
            raise KeyError(f"constituent {c!r} not tabulated; available: {CONSTITUENTS}")
        f[:, k] = np.hypot(t1, t2)
        u[:, k] = np.arctan2(t1, t2)
    return f, u


def predict(t: np.ndarray, hc: np.ndarray, constituents: list[str]) -> np.ndarray:
    """
    Tidal time series from complex harmonic constants (OTPS convention).

    Parameters
    ----------
    t : np.ndarray of datetime64, shape (nt,)
        Times (UT).
    hc : np.ndarray, complex, shape (..., nc)
        Harmonic constants h = Re + i·Im (TPXO hRe, hIm) for each constituent;
        leading dimensions are spatial (points, or (ny, nx)).
    constituents : list of str
        Names matching the last axis of ``hc``.

    Returns
    -------
    np.ndarray, shape (nt, ...)
        Predicted field, in the units of ``hc``.
    """
    hc = np.asarray(hc)
    if hc.shape[-1] != len(constituents):
        raise ValueError(f"hc has {hc.shape[-1]} values on its last axis but {len(constituents)} constituents")
    unknown = [c for c in constituents if c not in OMEGA]
    if unknown:
        raise KeyError(f"constituents {unknown} not tabulated; available: {CONSTITUENTS}")

    t = np.atleast_1d(np.asarray(t, dtype="datetime64[ns]"))
    mjd = to_mjd(t)
    f, u = nodal_corrections(mjd, constituents)
    seconds = (mjd - MJD_1992) * 86400.0
    omega = np.array([OMEGA[c] for c in constituents])
    phase = np.array([PHASE_1992[c] for c in constituents])
    theta = omega[None, :] * seconds[:, None] + phase[None, :] + u           # (nt, nc)

    space = hc.shape[:-1]
    h = hc.reshape(-1, len(constituents))                                     # (npts, nc)
    eta = (f * np.cos(theta)) @ h.real.T - (f * np.sin(theta)) @ h.imag.T     # (nt, npts)
    return eta.reshape((t.size,) + space)


def _interp_complex(field: "xr.DataArray", lat_name: str, lon_name: str,
                    lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a complex (constituent, lat, lon) field to points → (npts, nc)."""
    from scipy.interpolate import RegularGridInterpolator

    la = field[lat_name].values
    lo = field[lon_name].values
    order_lo, order_la = np.argsort(lo), np.argsort(la)
    data = field.values[:, order_la][:, :, order_lo]
    pts = np.column_stack([lat, lon])
    out = []
    for k in range(data.shape[0]):
        re = RegularGridInterpolator((la[order_la], lo[order_lo]), data[k].real, bounds_error=True)(pts)
        im = RegularGridInterpolator((la[order_la], lo[order_lo]), data[k].imag, bounds_error=True)(pts)
        out.append(re + 1j * im)
    return np.array(out).T


def boundary_tides(
    constants: "xr.Dataset",
    h_grid: "xr.Dataset",
    v_grid: "xr.Dataset",
    hfac: "xr.Dataset",
    flather_u: np.ndarray,
    flather_v: np.ndarray,
    t: np.datetime64,
    constituents: list[str] | None = None,
    ramp: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    Tidal elevation and depth-mean normal velocity at the Flather faces.

    Constants (``forcing.sources.tpxo.read_tpxo``) are interpolated bilinearly
    to each open face — η from the z nodes, transports from their own C-grid
    points — and predicted at time ``t`` with ``predict``.  The normal
    velocity is the TPXO transport divided by the *model* depth at the face,
    so that the tidal volume flux is conserved on the model bathymetry
    (rather than TPXO's velocity, which is transport over TPXO's depth).

    Parameters
    ----------
    constants : xr.Dataset
        ``h`` and, for velocities, ``U``, ``V`` from ``read_tpxo``.
    h_grid, v_grid, hfac : xr.Dataset
        Model grids and open-boundary hFac (``open_boundary_hfac``).
    flather_u, flather_v : np.ndarray of bool
        Flather faces, shapes (ny, nx+1) and (ny+1, nx).
    t : np.datetime64
        Time (for θ = ½ use the mid-step time).
    constituents : list of str, optional
        Subset to use; default all in ``constants``.
    ramp : float
        Multiplier in [0, 1] for spin-up, e.g. 1 − exp(−t/τ) (HYCOM ``tidrmp``).

    Returns
    -------
    dict
        ``eta_u``, ``ubar_u`` (ny, nx+1) and ``eta_v``, ``vbar_v`` (ny+1, nx),
        zero except on the Flather faces; ready for
        ``ImplicitFreeSurface.step(flather_eta=(eta_u, eta_v), …)``.
    """
    names = list(constants["constituent"].values) if constituents is None else list(constituents)
    c = constants.sel(constituent=names)
    dz = v_grid["dz"].values[:, None, None]
    out = {}
    for axis, mask, trans, lat_n, lon_n in (("u", flather_u, "U", "lat", "lon_u"),
                                           ("v", flather_v, "V", "lat_v", "lon")):
        mask = np.asarray(mask, bool)
        eta_f = np.zeros(mask.shape)
        vel_f = np.zeros(mask.shape)
        if mask.any():
            lat = h_grid[f"lat_{axis}"].values[mask]
            lon = h_grid[f"lon_{axis}"].values[mask]
            eta_f[mask] = ramp * predict(np.array([t]), _interp_complex(c["h"], "lat", "lon", lat, lon), names)[0]
            if trans in c:
                depth = np.sum(dz * hfac[f"hfac_{axis}"].values, axis=0)[mask]
                q = predict(np.array([t]), _interp_complex(c[trans], lat_n, lon_n, lat, lon), names)[0]
                vel_f[mask] = ramp * q / depth
        out[f"eta_{axis}"] = eta_f
        out["ubar_u" if axis == "u" else "vbar_v"] = vel_f
    return out
