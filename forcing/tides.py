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

The eight constituents HYCOM can force (``tidcon``) are tabulated in HYCOM
order (``CONSTITUENTS``), verified against pyTMD 2.1.7 (OTIS mode) to
round-off in ``tests/test_tides.py``.  The long-period MF and MM (regional
chain Task 5; NWA12 forces ten constituents, ``OBC_TIDE_CONSTITUENTS =
M2,S2,N2,K2,K1,O1,P1,Q1,MM,MF``, docs/design/regional_refs/mom6_obc_tides.md
§1) are added after them (``ALL_CONSTITUENTS``): ω and the astronomical
argument from OTPS ``constit.h`` (equal to the Doodson arguments 2s and
s − p of ``mean_longitudes`` at 1992-01-01 to 1e-9 rad, checked in
``tests/test_tides.py``), f and u from the long-period branch of the OTPS
``nodal`` routine (Schureman 1958): the first-order terms equal MOM6 nodal_fu
(MOM_tidal_forcing.F90:221-226); the 2N, 3N terms of MF u are Schureman's
[verificar against pyTMD ``arguments(..., corrections='OTIS')``: pyTMD is not
installed here; reference arrays in ``tests/data/tide_mm_mf_reference.npz`` come
from ``validation/tides/make_mm_mf_reference.py``].

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

#: Long-period constituents added for regional runs (NWA12 forces MM and MF at its open boundaries).
LONG_PERIOD: tuple[str, ...] = ("mf", "mm")

#: Every tabulated constituent: the HYCOM eight, then MF and MM.
ALL_CONSTITUENTS: tuple[str, ...] = CONSTITUENTS + LONG_PERIOD

#: Angular frequencies [rad s⁻¹] (OTPS constit.h).  MF = 2 ds/dt (13.661 d), MM = d(s − p)/dt (27.555 d);
#: MOM6 tidal_frequency (MOM_tidal_forcing.F90:154-182) has the same values.
OMEGA: dict[str, float] = {
    "m2": 1.405189e-04, "s2": 1.454441e-04, "k1": 7.292117e-05, "o1": 6.759774e-05,
    "n2": 1.378797e-04, "p1": 7.252295e-05, "k2": 1.458423e-04, "q1": 6.495854e-05,
    "mf": 0.053234e-04, "mm": 0.026392e-04,
}

#: Astronomical argument at 1992-01-01 00:00 UT [rad] (OTPS constit.h); MF = 2s, MM = s − p (Doodson; MOM6
#: eq_phase, MOM_tidal_forcing.F90:120-150, after Kowalik & Luick 2019 Table I.4).
PHASE_1992: dict[str, float] = {
    "m2": 1.731557546, "s2": 0.000000000, "k1": 0.173003674, "o1": 1.558553872,
    "n2": 6.050721243, "p1": 6.110181633, "k2": 3.487600001, "q1": 5.877717569,
    "mf": 1.756042456, "mm": 1.964021610,
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
        MF     : f = 1.043 + 0.414 cos N,  u = −23.7° sin N + 2.7° sin 2N − 0.4° sin 3N
        MM     : f = 1 − 0.130 cos N,      u = 0

    MF and MM are the long-period branch of the OTPS ``nodal`` routine (Schureman 1958; regional-chain Task 5).
    MOM6 uses only the first-order terms (MF f = 1.043 + 0.414 cos N, u = −23.7° sin N; MM the same f, u = 0;
    MOM_tidal_forcing.F90:187-231) and fixes them at TIDE_NODAL_REF_DATE; pyOCEAN evaluates them at each time
    (class c, docs/design/regional_refs/mom6_obc_tides.md §3).

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
        elif c == "mf":
            f[:, k] = 1.043 + 0.414 * cn
            u[:, k] = np.deg2rad(-23.7 * sn + 2.7 * s2n - 0.4 * s3n)
            continue
        elif c == "mm":
            f[:, k] = 1.0 - 0.130 * cn
            u[:, k] = 0.0
            continue
        else:
            raise KeyError(f"constituent {c!r} not tabulated; available: {ALL_CONSTITUENTS}")
        f[:, k] = np.hypot(t1, t2)
        u[:, k] = np.arctan2(t1, t2)
    return f, u


def predict(t: np.ndarray, hc: np.ndarray, constituents: list[str],
            nodal_mjd: float | None = None) -> np.ndarray:
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
    nodal_mjd : float, optional
        Evaluate f and u once at this Modified Julian Day instead of at each time (MOM6
        TIDE_NODAL_REF_DATE, MOM_open_boundary.F90:1599-1603, 1637-1642).  None (default): OTPS, at each t.

    Returns
    -------
    np.ndarray, shape (nt, ...)
        Predicted field, in the units of ``hc``.
    """
    t = np.atleast_1d(np.asarray(t, dtype="datetime64[ns]"))
    return predict_mjd(to_mjd(t), hc, constituents, nodal_mjd=nodal_mjd)


def predict_mjd(mjd: np.ndarray, hc: np.ndarray, constituents: list[str],
                nodal_mjd: float | None = None) -> np.ndarray:
    """
    ``predict`` at Modified Julian Days ``mjd`` (shape (nt,), float) instead of datetime64 — for model clocks
    in seconds, mjd = MJD(ref_date) + t/86400, without rounding t to an integer number of ns.  Same arithmetic.
    """
    hc = np.asarray(hc)
    if hc.shape[-1] != len(constituents):
        raise ValueError(f"hc has {hc.shape[-1]} values on its last axis but {len(constituents)} constituents")
    unknown = [c for c in constituents if c not in OMEGA]
    if unknown:
        raise KeyError(f"constituents {unknown} not tabulated; available: {ALL_CONSTITUENTS}")

    mjd = np.atleast_1d(np.asarray(mjd, float))
    if nodal_mjd is None:
        f, u = nodal_corrections(mjd, constituents)
    else:
        f, u = nodal_corrections(np.array([float(nodal_mjd)]), constituents)
    seconds = (mjd - MJD_1992) * 86400.0
    omega = np.array([OMEGA[c] for c in constituents])
    phase = np.array([PHASE_1992[c] for c in constituents])
    theta = omega[None, :] * seconds[:, None] + phase[None, :] + u           # (nt, nc)

    space = hc.shape[:-1]
    h = hc.reshape(-1, len(constituents))                                     # (npts, nc)
    eta = (f * np.cos(theta)) @ h.real.T - (f * np.sin(theta)) @ h.imag.T     # (nt, npts)
    return eta.reshape((mjd.size,) + space)


def _interp_complex(field: "xr.DataArray", lat_name: str, lon_name: str,
                    lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Land-aware bilinear interpolation of a complex (constituent, lat, lon) field.

    NaN (land) nodes are excluded and the weights of the remaining ocean
    nodes renormalised, ψ = Σ w_k ψ_k / Σ w_k over ocean nodes, so coastal
    values are not pulled towards zero (as in OTPS/pyTMD).  Points with no
    ocean node in their cell are NaN.  Returns (npts, nc).
    """
    from scipy.interpolate import RegularGridInterpolator

    la = field[lat_name].values
    lo = field[lon_name].values
    order_lo, order_la = np.argsort(lo), np.argsort(la)
    data = field.values[:, order_la][:, :, order_lo]
    pts = np.column_stack([lat, lon])
    axes = (la[order_la], lo[order_lo])
    out = []
    for k in range(data.shape[0]):
        ocean = np.isfinite(data[k])
        filled = np.where(ocean, data[k], 0.0)
        weight = RegularGridInterpolator(axes, ocean.astype(float), bounds_error=True)(pts)
        re = RegularGridInterpolator(axes, filled.real, bounds_error=True)(pts)
        im = RegularGridInterpolator(axes, filled.imag, bounds_error=True)(pts)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(weight > 1e-12, (re + 1j * im) / weight, np.nan))
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
    return boundary_tides_at(boundary_tide_constants(constants, h_grid, v_grid, hfac, flather_u, flather_v,
                                                     constituents), t, ramp=ramp)


def boundary_tide_constants(
    constants: "xr.Dataset",
    h_grid: "xr.Dataset",
    v_grid: "xr.Dataset",
    hfac: "xr.Dataset",
    flather_u: np.ndarray,
    flather_v: np.ndarray,
    constituents: list[str] | None = None,
) -> dict:
    """
    Harmonic constants of η and of the depth-mean normal velocity at the
    Flather faces, interpolated once (``boundary_tides`` does this every
    call).  Pass the result to ``boundary_tides_at`` each time step.

    Faces where the atlas has no ocean node take the nearest valid face's
    constants (``_fill_from_nearest``).

    Returns
    -------
    dict with ``names`` and, per face type (u, v): the mask, complex η
    constants (nf, nc) [m] and (transport constants (nf, nc) [m² s⁻¹], model
    depth (nf,) [m]) — divided after prediction, as in ``boundary_tides`` —
    or None without transports.
    """
    names = list(constants["constituent"].values) if constituents is None else list(constituents)
    c = constants.sel(constituent=names)
    dz = v_grid["dz"].values[:, None, None]
    out = {"names": names}
    for axis, mask, trans, lat_n, lon_n in (("u", flather_u, "U", "lat", "lon_u"),
                                           ("v", flather_v, "V", "lat_v", "lon")):
        mask = np.asarray(mask, bool)
        hc_eta = hc_vel = None
        if mask.any():
            lat = h_grid[f"lat_{axis}"].values[mask]
            lon = h_grid[f"lon_{axis}"].values[mask]
            hc_eta = _fill_from_nearest(_interp_complex(c["h"], "lat", "lon", lat, lon), lat, lon)
            if trans in c:
                depth = np.sum(dz * hfac[f"hfac_{axis}"].values, axis=0)[mask]
                hc_vel = (_fill_from_nearest(_interp_complex(c[trans], lat_n, lon_n, lat, lon), lat, lon), depth)
        out[axis] = (mask, hc_eta, hc_vel)
    return out


def _fill_from_nearest(hc: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Faces whose atlas cell holds only land (a model coastline finer than the
    atlas) take the constants of the nearest face with valid constants.
    """
    bad = ~np.all(np.isfinite(hc), axis=1)
    if bad.any() and (~bad).any():
        good = np.flatnonzero(~bad)
        coslat = np.cos(np.deg2rad(lat))
        for k in np.flatnonzero(bad):
            d = np.hypot((lon[good] - lon[k]) * coslat[k], lat[good] - lat[k])
            hc[k] = hc[good[np.argmin(d)]]
    return hc


def boundary_tides_at(bc: dict, t: np.datetime64, ramp: float = 1.0) -> dict[str, np.ndarray]:
    """
    Boundary tide at time ``t`` from ``boundary_tide_constants`` — the same
    output as ``boundary_tides`` without re-interpolating the atlas.
    """
    out = {}
    for axis in ("u", "v"):
        mask, hc_eta, hc_vel = bc[axis]
        eta_f = np.zeros(mask.shape)
        vel_f = np.zeros(mask.shape)
        if hc_eta is not None:
            eta_f[mask] = ramp * predict(np.array([t]), hc_eta, bc["names"])[0]
            if hc_vel is not None:
                q, depth = hc_vel
                vel_f[mask] = ramp * predict(np.array([t]), q, bc["names"])[0] / depth
        out[f"eta_{axis}"] = eta_f
        out["ubar_u" if axis == "u" else "vbar_v"] = vel_f
    return out


# ---------------------------------------------------------------------------
# Tidal potential (body tide) and self-attraction and loading (SAL)
# ---------------------------------------------------------------------------

#: Equilibrium-tide amplitudes A_c [m] (OTPS constit.h; Cartwright & Tayler 1971).
EQ_AMPLITUDE: dict[str, float] = {
    "m2": 0.2441, "s2": 0.112743, "k1": 0.141565, "o1": 0.100661,
    "n2": 0.046397, "p1": 0.046848, "k2": 0.030684, "q1": 0.019273,
    "mf": 0.042041, "mm": 0.022191,                       # OTPS constit.h = MOM6 MOM_tidal_forcing.F90:452, 457
}

#: α_c = 1 + k_c − h_c, Love-number factor of the body tide (Foreman et al. 1993; HYCOM mod_tides.F90).
LOVE_ALPHA: dict[str, float] = {
    "m2": 1 + 0.302 - 0.609, "s2": 1 + 0.302 - 0.609, "n2": 1 + 0.302 - 0.609, "k2": 1 + 0.302 - 0.609,
    "k1": 1 + 0.256 - 0.520, "o1": 1 + 0.298 - 0.603, "p1": 1 + 0.287 - 0.581, "q1": 1 + 0.298 - 0.603,
    # long period: 1 + k − h = 1 + 0.299 − 0.606 = 0.693 (OTPS constit.h alpha; MOM6 love_no 0.693,
    # MOM_tidal_forcing.F90:411-459)
    "mf": 1 + 0.299 - 0.606, "mm": 1 + 0.299 - 0.606,
}

#: Tidal species m (2 = semidiurnal, 1 = diurnal, 0 = long period).
SPECIES: dict[str, int] = {"m2": 2, "s2": 2, "n2": 2, "k2": 2, "k1": 1, "o1": 1, "p1": 1, "q1": 1,
                           "mf": 0, "mm": 0}

_G_TIDE: float = 9.81


def equilibrium_constants(lon: np.ndarray, lat: np.ndarray, constituents: list[str],
                          love: bool = True) -> np.ndarray:
    """
    Complex constants of the equilibrium tide, usable with ``predict``.

        semidiurnal:  h_c = α_c A_c cos²φ e^{2iλ}
        diurnal:      h_c = α_c A_c sin 2φ e^{iλ}
        long period:  h_c = α_c A_c (½ − 3/2 sin²φ)        (MF, MM; no longitude dependence)

    so that predict gives η_eq = α A f G(φ) cos(θ_c + mλ), as in HYCOM
    (mod_tides.F90: atide = A α cos²φ cos 2λ, btide = A α cos²φ sin 2λ).  The long-period structure
    ½ − 3/2 sin²φ with the argument V₀ = 2s (MF), s − p (MM) is MOM6's (tidal_forcing_init, struct = 3,
    MOM_tidal_forcing.F90:288-296, 451-458); sign checked in the Task 5 review (P₂(sin δ) gives a positive
    coefficient on cos 2s for MF and on cos(s − p) for MM).

    Parameters
    ----------
    lon, lat : np.ndarray
        Longitude [°E] and latitude [°N], same shape.
    constituents : list of str
    love : bool
        Include α = 1 + k − h (default) — the forcing felt by the ocean on a
        yielding Earth; False gives the astronomical equilibrium tide.

    Returns
    -------
    np.ndarray, complex, shape lon.shape + (nc,)
    """
    lam = np.deg2rad(np.asarray(lon, dtype=float))[..., None]
    phi = np.deg2rad(np.asarray(lat, dtype=float))[..., None]
    out = []
    for c in constituents:
        if c not in SPECIES:
            raise KeyError(f"no equilibrium tide for {c!r}; available: {ALL_CONSTITUENTS}")
        m = SPECIES[c]
        if m == 2:
            shape = np.cos(phi) ** 2
        elif m == 1:
            shape = np.sin(2.0 * phi)
        else:
            shape = 0.5 - 1.5 * np.sin(phi) ** 2                          # long period: no λ
        alpha = LOVE_ALPHA[c] if love else 1.0
        out.append(alpha * EQ_AMPLITUDE[c] * shape * np.exp(1j * m * lam))
    return np.concatenate(out, axis=-1)


def body_tide(lon: np.ndarray, lat: np.ndarray, t: np.ndarray, constituents: list[str],
              sal: np.ndarray | None = None, ramp: float = 1.0) -> np.ndarray:
    """
    Effective tidal forcing height η_body = Σ_c [α_c η_eq,c − η_SAL,c]  [m].

    The ocean is forced by −∇(−g η_body) (HYCOM ``etide``, tidflg ≥ 2).

    Parameters
    ----------
    lon, lat : np.ndarray
        Coordinates of the points (e.g. model T-points).
    t : np.ndarray of datetime64
    constituents : list of str
    sal : np.ndarray, complex, shape lon.shape + (nc,), optional
        Per-constituent SAL constants (``read_sal_hycom``), subtracted.
    ramp : float
        Spin-up factor (HYCOM ``tidrmp``: 1 − exp(−5 t/τ)).

    Returns
    -------
    np.ndarray, shape (nt,) + lon.shape
    """
    hc = equilibrium_constants(lon, lat, constituents)
    if sal is not None:
        hc = hc - sal
    return ramp * predict(t, hc, constituents)


def body_force(h_grid: "xr.Dataset", eta_body: np.ndarray, g: float = _G_TIDE) -> tuple[np.ndarray, np.ndarray]:
    """
    Tidal body-force acceleration g ∇η_body at U- and V-faces [m s⁻²].

        F_x[j, i] = g (η_b[j, i] − η_b[j, i−1]) / Δx_u,   F_y likewise;  0 on the domain edges

    Add it to the momentum tendency of ``ab2_step_implicit_fs``.
    """
    ny, nx = eta_body.shape
    gx = np.zeros((ny, nx + 1))
    gy = np.zeros((ny + 1, nx))
    gx[:, 1:-1] = g * (eta_body[:, 1:] - eta_body[:, :-1]) / h_grid["dx_u"].values[:, 1:-1]
    gy[1:-1, :] = g * (eta_body[1:, :] - eta_body[:-1, :]) / h_grid["dy_v"].values[1:-1, :]
    return gx, gy


def read_sal_hycom(path_b, idm: int | None = None, jdm: int | None = None,
                   constituents: tuple[str, ...] = CONSTITUENTS) -> np.ndarray:
    """
    HYCOM ``tidal.salReIm.[ab]`` → complex SAL constants, shape (jdm, idm, nc) [m].

    Records are Re, Im of each constituent in HYCOM order (mod_tides.F90,
    tides_forfun_sal), as produced for HYCOM-IOA by calcSAL + sal2hycom.py.

    Sign: HYCOM subtracts the file, etide = α η_eq − F, so F must hold
    −η_SAL.  The real PAC_MEX file does: F ≈ −0.10 η (M2, S2, N2, K2) and
    ≈ −0.14 η (diurnals) against TPXO10 over deep water, correlation 0.95–0.99
    (validation/tides/check_sal_real.py).  ``body_tide(sal=F)`` uses the same
    convention.
    """
    from forcing.sources.hycom_ab import read_record

    out = []
    for k in range(len(constituents)):
        re = read_record(path_b, 2 * k, idm, jdm)
        im = read_record(path_b, 2 * k + 1, idm, jdm)
        out.append(re + 1j * im)
    return np.stack(out, axis=-1)
