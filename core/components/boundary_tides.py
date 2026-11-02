"""
Tides at the open boundaries of the 3-D core and the tidal potential (body tide) in 3-D (regional chain Task 5).

``BoundaryTides`` — barotropic tide added to the Flather exterior of the ``'radiating'`` open edges:

    η_ext ← η_ext + r(t) η_tide(t),        ū_ext ← ū_ext + r(t) ū_tide(t)

with η_tide, ū_tide predicted from TPXO harmonic constants with the OTPS conventions
(``forcing.tides.predict_mjd``; Egbert & Erofeeva 2002): η from the z nodes, the normal velocity from the TPXO
transports U (east), V (north) interpolated to the face, rotated to grid axes with ``forcing.rotation.to_grid``
when the grid carries ``cos_t``/``sin_t`` (curvilinear supergrid grids), and divided by the **model** rest depth of
the face, u = U/D, so that the tidal volume flux is that of TPXO on the model bathymetry (ruling R3; MOM6/NWA12 use
TPXO's velocities instead, MOM_open_boundary.F90:4884-4893 with CEFI write_tpxo_boundary.py:32,43 — class c).
The tidal velocity enters the depth mean ū of the Flather condition only; the baroclinic exterior u′ is untouched.
MOM6 adds the tidal velocity to every layer (normal_vel(k) = u_GLORYS(k) + tidal_vn, MOM_open_boundary.F90:4900,
4912), a depth-uniform increment, which changes only the depth mean ``normal_vel_bt`` that its Flather uses: the
same thing (it also enters MOM6's nudging target of the layer velocities, where pyOCEAN's u′ target is the
provider's; docs/design/regional_refs/mom6_obc_tides.md §3, §8 item 2).

Nodal corrections f, u are evaluated at each time (OTPS) by default; ``nodal_ref_date`` fixes them at that date as
MOM6's TIDE_NODAL_REF_DATE does (MOM_open_boundary.F90:1599-1603, 1637-1642; NWA12 1993-07-02).  The phase is the
OTPS astronomical argument at the actual time, so ``ref_date`` is only the model's time origin: model time t [s]
is the date ref_date + t.  ``ramp_days`` > 0 multiplies the tide (only the tide; MOM6's RAMP_OBCS multiplies the
whole SSH, MOM_open_boundary.F90:4980-4983, and NWA12 has it off) by r = min(t / (ramp_days · 86400), 1), linear in
the model time from t = 0, so it is a pure function of t and a restart needs no state; default 0: r = 1.

The tide is evaluated at tⁿ⁺¹, the time level of the time-varying exterior ``open_boundary_state``
(core.components.open_boundary) and of MOM6, whose OBC tides are updated with Time_local = tⁿ⁺¹
(update_OBC_data, MOM_dynamics_split_RK2.F90:315, 553; ruling M1 of the Task 5 review).  The 2-D PAC_MEX runner
uses tⁿ + Δt/2 (validation/pacmex_tides/run_tides.py:299-337), a phase ωΔt/2 earlier (2.4° of M2 at Δt = 600 s);
tⁿ + θΔt would match the θ-weighted Flather flux best — all three are class (c) choices.  The body tide stays at
the mid-step tⁿ + Δt/2, as MOM6 (calc_tidal_forcing(CS%Time), MOM_PressureForce_FV.F90:1179, CS%Time at mid-step,
MOM.F90:877).

``TidalPotential`` — the equilibrium tide as a body force on every layer, −∇(−g η_body) = g ∇η_body with
η_body = Σ_c α_c η_eq,c (``forcing.tides.equilibrium_constants``, α = 1 + k − h; HYCOM mod_tides.F90, MOM6
calc_tidal_forcing, MOM_tidal_forcing.F90:638-646), evaluated at tⁿ + Δt/2 and added to the momentum tendency after
the AB3 combination (time-centred, not extrapolated, as the wind stress).  With the scalar self-attraction and
loading of the free surface (``sal_beta``, g(1 − β)∇η on the total η, ``core.dynamics.wetting_drying``) the surface
pressure gradient becomes g∇[(1 − β)η − η_body], the barotropic form of MOM6's legacy path
(MOM_PressureForce_FV.F90:1185-1196); MOM6's shift of every interface before the density integrals is not
reproduced (class c/d, mom6_obc_tides.md §6).

Index order (k, j, i), k = 0 at the bed; u-faces (ny, nx+1), v-faces (ny+1, nx).
"""
from __future__ import annotations

import copy

import numpy as np

from forcing.rotation import to_grid
from forcing.tides import (OMEGA, SPECIES, _fill_from_nearest, _interp_complex, body_force, equilibrium_constants,
                           predict_mjd, to_mjd)

EDGE_FACES = {"west": ("u", np.s_[:, 0], np.s_[:, 0]), "east": ("u", np.s_[:, -1], np.s_[:, -1]),
              "south": ("v", np.s_[0, :], np.s_[0, :]), "north": ("v", np.s_[-1, :], np.s_[-1, :])}


def _date(d) -> np.datetime64:
    return np.datetime64(d, "ns")


class _TideClock:
    """Model time t [s] → Modified Julian Day, fixed nodal date and linear ramp (shared by both classes)."""

    def __init__(self, constituents, ref_date, nodal_ref_date=None, ramp_days: float = 0.0) -> None:
        self.constituents = [str(c).lower() for c in constituents]
        self.ref_date = _date(ref_date)
        self.nodal_ref_date = None if nodal_ref_date is None else _date(nodal_ref_date)
        self.ramp_days = float(ramp_days)
        self._mjd0 = float(to_mjd(self.ref_date))
        self._nodal_mjd = None if self.nodal_ref_date is None else float(to_mjd(self.nodal_ref_date))

    def mjd(self, t: float) -> float:
        """Modified Julian Day of model time t [s] (t = 0 at ref_date)."""
        return self._mjd0 + float(t) / 86400.0

    def ramp(self, t: float) -> float:
        """r(t) = min(t / (ramp_days · 86400), 1), 1 without a ramp."""
        if self.ramp_days <= 0.0:
            return 1.0
        return float(min(max(float(t), 0.0) / (self.ramp_days * 86400.0), 1.0))

    def issues(self, name: str) -> list[str]:
        """Configuration errors (pre-flight)."""
        out = []
        unknown = [c for c in self.constituents if c not in OMEGA]
        if unknown:
            out.append(f"{name}: constituents {unknown} are not tabulated in forcing.tides (have {sorted(OMEGA)})")
        if not self.constituents:
            out.append(f"{name}: no constituents")
        if not (np.isfinite(self.ramp_days) and self.ramp_days >= 0.0):
            out.append(f"{name}: ramp_days={self.ramp_days!r} must be a finite number ≥ 0 (0 = no ramp)")
        if np.isnat(self.ref_date):
            out.append(f"{name}: ref_date is not a date")
        return out

    def config(self) -> dict[str, str | float]:
        """Values a restart must match (the clock has no state: everything is a function of t)."""
        return dict(constituents=",".join(self.constituents), ref_date=str(self.ref_date),
                    nodal_ref_date="none" if self.nodal_ref_date is None else str(self.nodal_ref_date),
                    ramp_days=self.ramp_days)


class BoundaryTides(_TideClock):
    """
    Barotropic tide at the Flather (``'radiating'``) open edges of ``SigmaBaroclinic`` (module docstring).

    Parameters
    ----------
    constants : xr.Dataset
        TPXO-style complex harmonic constants (``forcing.sources.tpxo.read_tpxo``): ``h`` (constituent, lat, lon)
        [m] and optionally ``U`` (constituent, lat, lon_u), ``V`` (constituent, lat_v, lon) transports [m² s⁻¹]
        (east, north).  Without ``U``/``V`` the tidal velocity is zero.  Land nodes NaN.
    constituents : list of str
        TPXO/OTPS names (``"m2"``, …, ``"mf"``, ``"mm"``); a subset of ``constants.constituent``.
    ref_date : np.datetime64
        Date of model time t = 0.
    nodal_ref_date : np.datetime64, optional
        Fix f, u at this date (MOM6 TIDE_NODAL_REF_DATE); None: OTPS, at each time.
    ramp_days : float
        Linear ramp of the tide over this many days from t = 0 (0: none, NWA12).
    velocity_depth : {'model', 'tpxo'}
        Depth that turns the TPXO transports into the tidal velocity: 'model' (default, ruling R3) u = U/D with the
        model's rest depth of the face; 'tpxo' u = U/h_TPXO, the atlas's own velocity (U/hu at the U nodes, V/hv at
        the V nodes, then interpolated to the face and rotated) — the velocity MOM6/NWA12 prescribe (CEFI
        write_tpxo_boundary.py), so the tidal volume flux follows the model depth.  Needs ``hu``/``hv`` in the
        constants (``read_tpxo`` with the atlas grid file).
    """

    def __init__(self, constants, constituents, ref_date, nodal_ref_date=None, ramp_days: float = 0.0,
                 velocity_depth: str = "model") -> None:
        super().__init__(constituents, ref_date, nodal_ref_date, ramp_days)
        self.velocity_depth = velocity_depth
        self.constants = constants
        self.edges: tuple[str, ...] = ()
        self._faces: dict[str, tuple] = {}

    def issues(self, name: str = "boundary_tides") -> list[str]:
        out = super().issues(name)
        have = [str(c).lower() for c in self.constants["constituent"].values] if "constituent" in self.constants \
            else []
        missing = [c for c in self.constituents if c not in have]
        if missing:
            out.append(f"{name}: constituents {missing} are not in the harmonic constants (have {have})")
        if "h" not in self.constants:
            out.append(f"{name}: the constants need the elevation 'h' (constituent, lat, lon)")
        if ("U" in self.constants) != ("V" in self.constants):
            out.append(f"{name}: give both transports 'U' and 'V' (or neither)")
        if self.velocity_depth not in ("model", "tpxo"):
            out.append(f"{name}: velocity_depth={self.velocity_depth!r}: use 'model' or 'tpxo'")
        elif self.velocity_depth == "tpxo" and not ("hu" in self.constants and "hv" in self.constants):
            out.append(f"{name}: velocity_depth='tpxo' needs the atlas depths 'hu' and 'hv' in the constants "
                       "(read_tpxo with the TPXO grid file)")
        return out

    def config(self) -> dict[str, str | float]:
        """Restart check values; ``velocity_depth`` only when it is not the default (old restarts stay valid)."""
        out = super().config()
        if self.velocity_depth != "model":
            out["velocity_depth"] = self.velocity_depth
        return out

    # ------------------------------------------------------------------ set-up
    def bind(self, h_grid, edges: tuple[str, ...], Hu: np.ndarray, Hv: np.ndarray) -> "BoundaryTides":
        """
        Interpolate the constants once to the faces of ``edges`` (land-aware bilinear, faces whose atlas cell is
        all land take the nearest valid face: ``forcing.tides._interp_complex``, ``_fill_from_nearest``) and
        convert transports to velocities with the rest face depths ``Hu`` (ny, nx+1), ``Hv`` (ny+1, nx) [m] (the
        boundary cell's depth at open edge faces, ``OpenBoundary.edge_face_depths``; 0 on land).
        """
        for k in ("lon_u", "lat_u", "lon_v", "lat_v"):
            if k not in h_grid:
                raise ValueError(f"boundary_tides need the face longitudes and latitudes of the grid ({k!r} missing): "
                                 "use a spherical or supergrid grid, or add lon_u/lat_u/lon_v/lat_v")
        names = self.constituents
        c = self.constants.assign_coords(constituent=[str(x).lower() for x in self.constants["constituent"].values])
        c = c.sel(constituent=names)
        rotate = "cos_t" in h_grid and "sin_t" in h_grid
        self.edges, self._faces = tuple(edges), {}
        self.shape_u, self.shape_v = Hu.shape, Hv.shape
        for e in self.edges:
            kind, sl_f, sl_c = EDGE_FACES[e]
            lon = np.asarray(h_grid[f"lon_{kind}"].values, float)[sl_f]
            lat = np.asarray(h_grid[f"lat_{kind}"].values, float)[sl_f]
            lon = ((lon + 180.0) % 360.0) - 180.0                       # TPXO constants are in −180 … 180 °E
            depth = np.nan_to_num(np.asarray(Hu if kind == "u" else Hv, float)[sl_f])
            ok = depth > 0.0
            hc_eta = np.zeros((lon.size, len(names)), complex)
            hc_vel = np.zeros_like(hc_eta)
            if ok.any():
                hc_eta[ok] = _fill_from_nearest(_interp_complex(c["h"], "lat", "lon", lat[ok], lon[ok]),
                                                lat[ok], lon[ok])
                if "U" in c and "V" in c:
                    cU, cV = c["U"], c["V"]
                    if self.velocity_depth == "tpxo":                   # the atlas's own velocities U/hu, V/hv
                        cU, cV = cU / c["hu"], cV / c["hv"]
                    Ue = _fill_from_nearest(_interp_complex(cU, "lat", "lon_u", lat[ok], lon[ok]), lat[ok], lon[ok])
                    Vn = _fill_from_nearest(_interp_complex(cV, "lat_v", "lon", lat[ok], lon[ok]), lat[ok], lon[ok])
                    if rotate:                                          # earth → grid axes (ruling R1 convention)
                        cs = np.asarray(h_grid["cos_t"].values, float)[sl_c][ok][:, None]
                        sn = np.asarray(h_grid["sin_t"].values, float)[sl_c][ok][:, None]
                        Ux, Vy = to_grid(Ue, Vn, cs, sn)
                    else:
                        Ux, Vy = Ue, Vn
                    if self.velocity_depth == "tpxo":
                        hc_vel[ok] = Ux if kind == "u" else Vy                       # already a velocity
                    else:
                        hc_vel[ok] = (Ux if kind == "u" else Vy) / depth[ok][:, None]   # u = U / D (model depth)
            if not (np.all(np.isfinite(hc_eta)) and np.all(np.isfinite(hc_vel))):
                raise ValueError(f"boundary_tides: edge {e!r} has faces without valid harmonic constants (the atlas "
                                 "holds only land around the whole edge)")
            self._faces[e] = (kind, sl_f, ok, hc_eta, hc_vel)
        return self

    # ------------------------------------------------------------------ time
    def at(self, t: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        {edge: (η_tide [m], ū_tide [m s⁻¹] normal to the edge, grid-x for west/east and grid-y for south/north,
        positive toward +x/+y)} along each bound edge at model time t [s], ramp included; 0 on land faces.
        """
        mjd, r = np.array([self.mjd(t)]), self.ramp(t)
        out = {}
        for e, (_, _, ok, hc_eta, hc_vel) in self._faces.items():
            both = predict_mjd(mjd, np.stack([hc_eta, hc_vel]), self.constituents, nodal_mjd=self._nodal_mjd)[0]
            out[e] = (np.where(ok, r * both[0], 0.0), np.where(ok, r * both[1], 0.0))
        return out

    def flather_arrays(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(η_u, η_v, ū_u, v̄_v) at all faces (zero away from the bound edges), the layout of the Flather targets."""
        eu, ev = np.zeros(self.shape_u), np.zeros(self.shape_v)
        uu, vv = np.zeros(self.shape_u), np.zeros(self.shape_v)
        for e, (eta, vel) in self.at(t).items():
            kind, sl_f = self._faces[e][:2]
            (eu if kind == "u" else ev)[sl_f] = eta
            (uu if kind == "u" else vv)[sl_f] = vel
        return eu, ev, uu, vv


class TidalPotential(_TideClock):
    """
    Equilibrium (body) tide as a momentum forcing g∇η_body [m s⁻²] at u- and v-faces (module docstring).

    Parameters as ``BoundaryTides`` (constituents need an equilibrium tide: ``forcing.tides.SPECIES``).
    ``SigmaBaroclinic(tidal_potential=True)`` builds it from the ``boundary_tides`` constituents and dates.
    """

    @classmethod
    def from_boundary_tides(cls, bt: BoundaryTides) -> "TidalPotential":
        return cls([c for c in bt.constituents if c in SPECIES], bt.ref_date, bt.nodal_ref_date, bt.ramp_days)

    def issues(self, name: str = "tidal_potential") -> list[str]:
        out = super().issues(name)
        noeq = [c for c in self.constituents if c in OMEGA and c not in SPECIES]
        if noeq:
            out.append(f"{name}: no equilibrium tide for {noeq} (have {sorted(SPECIES)})")
        return out

    def bind(self, h_grid, g: float, active_u: np.ndarray, active_v: np.ndarray, per=None) -> "TidalPotential":
        """
        Equilibrium constants at the T points (``lon_t``, ``lat_t``), the face spacings and the active faces
        (``H_u``, ``H_v`` finite); periodic axes get the wrap-face gradient (``per``: core.model.periodic).
        """
        for k in ("lon_t", "lat_t"):
            if k not in h_grid:
                raise ValueError(f"tidal_potential needs the cell longitudes and latitudes of the grid ({k!r} missing)")
        self.hg, self.g = h_grid, float(g)
        self._hc = equilibrium_constants(h_grid["lon_t"].values, h_grid["lat_t"].values, self.constituents)
        self._act_u, self._act_v = np.asarray(active_u, bool), np.asarray(active_v, bool)
        self._per = per
        return self

    def eta_body(self, t: float) -> np.ndarray:
        """η_body = r(t) Σ α η_eq [m] at the T points, model time t [s]."""
        return self.ramp(t) * predict_mjd(np.array([self.mjd(t)]), self._hc, self.constituents,
                                          nodal_mjd=self._nodal_mjd)[0]

    def force(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        (F_x, F_y) = g ∇η_body at u- and v-faces [m s⁻²] (``forcing.tides.body_force``: centred differences over
        Δx_u, Δy_v; zero on domain-edge faces and inactive faces; the periodic wrap face from its two cells).
        """
        eb = self.eta_body(t)
        gx, gy = body_force(self.hg, eb, g=self.g)
        per = self._per
        if per is not None and getattr(per, "x", False):
            gx[:, 0] = gx[:, -1] = self.g * (eb[:, 0] - eb[:, -1]) / self.hg["dx_u"].values[:, 0]
        if per is not None and getattr(per, "y", False):
            gy[0, :] = gy[-1, :] = self.g * (eb[0, :] - eb[-1, :]) / self.hg["dy_v"].values[0, :]
        return np.where(self._act_u, gx, 0.0), np.where(self._act_v, gy, 0.0)


def bind_copy(obj, *args, **kw):
    """A shallow copy of ``obj`` bound to this model (the caller's object stays unbound and reusable)."""
    return copy.copy(obj).bind(*args, **kw)
