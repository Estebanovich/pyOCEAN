"""
Bulk surface fluxes as a component of the 3-D model (N2b task 4, spec
docs/superpowers/specs/2026-10-31-n2b-bulk-surface-fluxes-design.md, decisions 1–9 of "Decisiones de Claude").

Each step the component turns the atmosphere at t^{n+½} and the ocean surface of the state n into the four forcing
arguments the model already takes (``SigmaBaroclinic.step``: wind stress, surface heat flux, freshwater flux):

1. **Surface state** (spec decision 3, anti-goal 1).  SST, SSS and the surface current are thickness-weighted means
   over the top ``h_sfc`` metres (MOM6 ``HMIX_SFC_PROP`` = 1 m, ``MOM.F90:2695, 4242-4301``: each layer adds
   dh = min(h_k, h_sfc − depth above it), SST = Σ dh T / Σ dh), the same rule in σ, z* and hybrid, so a 1 mm or a
   200 m top layer does not set the SST.  A column shallower than ``h_sfc`` averages the whole column (MOM6 divides
   by the depth reached).  The current uses ``h_uv_sfc`` (None = ``h_sfc``; 0 = the top layer, MOM6's default
   ``HMIX_UV_SFC_PROP = 0``, ``MOM.F90:2700-2704``): pyOCEAN averages it like the SST by default, a documented
   choice (c) so the relative wind does not depend on the top-layer thickness.  The layer velocities are first
   taken to the T points, ½(u_{i−½} + u_{i+½}) per layer.
2. **Atmosphere** at t^{n+½} = t^n + Δt/2 (decision 2) from ``forcing.atmosphere.AtmosphereProvider`` (canonical
   fields on the T points: u10, v10 [m s⁻¹], t2m [K], q2m [kg kg⁻¹], msl [Pa], sw_down, lw_down [W m⁻²], precip
   [m s⁻¹ of liquid water]).  The ocean state is that of the start of the step, the explicit coupling of one
   coupler step; every phase of the step uses the same fluxes.
3. **Bulk formula** (``scheme``):

   ``'ncar'``       ``forcing.bulk.bulk_ncar_fms`` (FMS coupler form of Large & Yeager 2004, used by MOM6/OM4); SST
                    in K, heights z_u, z_t, z_q.
   ``'ncar_ly04'``  ``forcing.bulk.bulk_ncar`` (LY04 as MITgcm ``pkg/exf``); SST in °C, one height z_t for T and q.
   ``'coare'``      ``forcing.bulk_coare.bulk_coare_croco`` (COARE 3.0 as CROCO ``bulk_flux.F``, git ab63b58), called
                    directly (controller ruling R2): t_air and SST in °C, the wind already relative to the current,
                    CROCO's constant p_sfc = 1e5 Pa (``PSURF``, CROCO without READ_PATM; ``msl`` is not used, as
                    in CROCO, so the comparison with CROCO is one of models and not of inputs), one height z_t.
   ``'kara'``       ``forcing.bulk.bulk_kara`` (HYCOM flxflg=6); SST in °C, SSS of the model; HYCOM uses T and q at
                    their measurement height without a height correction, so z_u, z_t, z_q are not used.

   Relative wind ΔU = U₁₀ − u_sfc in every scheme (``relative_wind=True``, LY04 and MOM6); ``False`` gives the
   absolute wind of CROCO, for the separation runs of the spec's criterion 5 only.
4. **Radiation** (decision 8).  Q_sw = (1 − albedo) SW↓ in every scheme (LY04 eq. 2, α = 0.066; for 'coare' CROCO
   expects the net shortwave from its preprocessing, ruling R2).  Longwave: ε (LW↓ − σ SST⁴) with ε = 0.97
   (``forcing.bulk.radiative_fluxes``; LY04, MITgcm ``exf_radiation.F``) for 'ncar', 'ncar_ly04' and 'kara';
   'coare' keeps CROCO's own LW↓ − 0.985 σ SST⁴ without reflection (``CoareFluxes.q_lw``, ``bulk_flux.F:617-618``,
   docs/design/n2b_refs/croco_bulk.md §4).
5. **To the model.**  heat = {"Q_ns": Q_sens + Q_lat + Q_lw, "Q_sw": Q_sw} (Q_sw penetrates with the Jerlov
   profile of N3, ``physics.surface.shortwave``); fw = P − E; τ at the u and v faces: the mean of the two wet
   neighbour centres, the wet value where one neighbour is land or dry, 0 where both are, wrapped on periodic axes
   (decision 7) — MOM6's A-grid stress, mask-weighted mean of the two T points
   (``MOM_surface_forcing_gfdl.F90:1086-1097``).  Land and dry columns (the surface-flux mask of the driver:
   active, H + ηⁿ > d_min, area > 0) get 0 in every field.

Freshwater volume convention (Task 2 review m5).  The model is Boussinesq: a mass flux M [kg m⁻² s⁻¹] changes the
layer thickness by M/ρ₀ with the model's ρ₀ (MOM6 ``GV%kg_m2_to_H = 1/(Rho0 H_to_m)``,
``MOM_verticalGrid.F90:195-197``, used on the surface mass fluxes in ``applyBoundaryFluxesInOut``; MITgcm
``mass2rUnit = 1/rhoConst``).  The schemes return E as a volume flux at different densities, so each is taken back
to its mass flux first:

    'ncar', 'ncar_ly04'  E = F_E/ρ_fw,    ρ_fw   = 1000  (``forcing.bulk.RHO_FW``)
    'coare'              E = −Q_L/(ρ₀ L_v), ρ₀    = 1025  (CROCO's rho0, ``RHO0_CROCO``, passed explicitly)
    'kara'               E = −Q_L/(L_v ρ_ref), ρ_ref = 1000 (HYCOM ``svref``, ``forcing.bulk._HY_RHOREF``)
    precip               liquid-water equivalent, ρ_fw = 1000 (ERA5 ``tp`` and WRF are metres of water)

and fw = (ρ_fw P − ρ_E E)/ρ₀_model [m s⁻¹].  ``last['evap']`` and ``last['precip']`` are in this model convention,
so fw = precip − evap holds exactly.

The component holds no state (decision 6): the provider is a pure function of time, so a restart only checks the
configuration (``core.restart``).

Index order (k, j, i), k = 0 at the bed: the top layer is k = N−1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from forcing.bulk import (RHO_FW, T0_K, _HY_RHOREF, bulk_kara, bulk_ncar, bulk_ncar_fms,  # noqa: F401
                          radiative_fluxes)
from forcing.bulk_coare import PSURF, RHO0_CROCO, bulk_coare_croco

#: Bulk schemes of ``BulkFluxes.scheme``.
SCHEMES: tuple[str, ...] = ("ncar", "ncar_ly04", "coare", "kara")

#: Density [kg m⁻³] at which each scheme's evaporation is a volume flux (module docstring).
EVAP_DENSITY: dict[str, float] = {"ncar": RHO_FW, "ncar_ly04": RHO_FW, "coare": RHO0_CROCO, "kara": _HY_RHOREF}

#: Fields of ``SurfaceFluxes.last`` (all (ny, nx) at the T points, 0 on land and dry cells; sst_bulk NaN there).
LAST_FIELDS: tuple[str, ...] = ("q_sens", "q_lat", "q_lw", "q_sw", "evap", "precip", "fw", "tau_x", "tau_y",
                                "sst_bulk")


@dataclass
class BulkFluxes:
    """
    Configuration of the bulk surface fluxes (``SigmaBaroclinic(..., surface_fluxes=BulkFluxes(...))``).

    atmosphere     ``forcing.atmosphere.AtmosphereProvider`` (or any object with ``__call__(t) -> dict`` of the
                   canonical fields, ``coverage() -> (t0, t1)`` [s] and ``origin``)
    scheme         'ncar' (``bulk_ncar_fms``, default), 'ncar_ly04' (``bulk_ncar``, MITgcm), 'coare'
                   (``bulk_coare_croco``, CROCO) or 'kara' (``bulk_kara``, HYCOM)
    h_sfc          depth of the SST/SSS average [m] (MOM6 ``HMIX_SFC_PROP`` = 1, ``MOM.F90:2695``); 0 = top layer
    h_uv_sfc       depth of the surface-current average [m]; None = ``h_sfc``; 0 = the top layer (MOM6
                   ``HMIX_UV_SFC_PROP``)
    relative_wind  wind relative to the surface current (True, LY04 and MOM6); False = CROCO's absolute wind
    albedo         shortwave albedo [–] (LY04 0.066; spec decision 8)
    emissivity     longwave emissivity [–] (LY04/MITgcm 0.97); 'coare' uses CROCO's 0.985 without reflection
    z_u, z_t, z_q  heights of the wind, temperature and humidity [m] (ERA5: 10, 2, 2)
    """

    atmosphere: Any
    scheme: str = "ncar"
    h_sfc: float = 1.0
    h_uv_sfc: float | None = None
    relative_wind: bool = True
    albedo: float = 0.066
    emissivity: float = 0.97
    z_u: float = 10.0
    z_t: float = 2.0
    z_q: float = 2.0


def bulk_errors(b: Any, dt: float, salinity: bool) -> list[str]:
    """
    Every problem of a ``BulkFluxes`` configuration, for the constructor pre-flight (spec criterion 6): the type,
    an unknown scheme, no salinity (E − P acts on S, and 'kara' needs the SSS), negative averaging depths, albedo
    outside [0, 1], emissivity outside (0, 1], z_q ≠ z_t for the one-height schemes ('ncar_ly04', 'coare'), and an
    atmosphere that does not cover [0, Δt/2], the time of the first step's fluxes.
    """
    if not isinstance(b, BulkFluxes):
        return [f"surface_fluxes must be a core.components.surface_fluxes.BulkFluxes, got {type(b).__name__}"]
    out = []
    if b.scheme not in SCHEMES:
        out.append(f"unknown bulk scheme {b.scheme!r} (valid: {', '.join(repr(s) for s in SCHEMES)})")
    if not salinity:
        out.append("surface_fluxes needs salinity=True (the freshwater flux P − E acts on the salinity S"
                   + (", and 'kara' needs the SSS)" if b.scheme == "kara" else ")"))
    for name in ("h_sfc", "h_uv_sfc"):
        v = getattr(b, name)
        if v is not None and not (np.isfinite(v) and v >= 0):
            out.append(f"surface_fluxes.{name}={v!r}: the averaging depth must be ≥ 0 m (0 = the top layer)")
    if not (np.isfinite(b.albedo) and 0.0 <= b.albedo <= 1.0):
        out.append(f"surface_fluxes.albedo={b.albedo!r} must be in [0, 1] (LY04: 0.066)")
    if not (np.isfinite(b.emissivity) and 0.0 < b.emissivity <= 1.0):
        out.append(f"surface_fluxes.emissivity={b.emissivity!r} must be in (0, 1] (LY04: 0.97)")
    for name in ("z_u", "z_t", "z_q"):
        v = getattr(b, name)
        if not (np.isfinite(v) and v > 0):
            out.append(f"surface_fluxes.{name}={v!r}: a measurement height must be > 0 m")
    if b.scheme in ("ncar_ly04", "coare") and b.z_q != b.z_t:
        out.append(f"surface_fluxes.z_q={b.z_q!r} ≠ z_t={b.z_t!r}: scheme {b.scheme!r} takes T and q at one "
                   "height (z_t)")
    try:
        c0, c1 = (float(c) for c in b.atmosphere.coverage())
    except Exception as e:                                             # noqa: BLE001 — reported, not raised
        out.append(f"surface_fluxes.atmosphere has no usable coverage() ({e})")
    else:
        if not (c0 <= 0.0 and c1 >= 0.5 * dt):
            out.append(f"the atmosphere covers [{c0}, {c1}] s from its origin and does not cover [0, {0.5 * dt}] s, "
                       "the time of the first step's fluxes")
    return out


def surface_average(h: np.ndarray, q: np.ndarray, h_sfc: float) -> np.ndarray:
    """
    Thickness-weighted mean of q (N, ny, nx) over the top ``h_sfc`` metres → (ny, nx), the MOM6 surface state
    (``MOM.F90:4242-4301``):

        w_k = clip(h_sfc − d_k, 0, h_k),   d_k = Σ_{j > k} h_j (depth of the top of layer k),
        q̄ = Σ w_k q_k / Σ w_k

    from the top layer k = N−1 down.  A column shallower than h_sfc gives the mean of the whole column;
    ``h_sfc == 0`` gives the top layer q_{N−1}; an empty column (Σ h = 0) gives NaN.
    """
    h = np.asarray(h, float)
    q = np.asarray(q, float)
    D = h.sum(axis=0)
    if h_sfc == 0:
        return np.where(D > 0, q[-1], np.nan)
    below = np.cumsum(h[::-1], axis=0)[::-1]                    # depth of the bottom of layer k
    w = np.clip(np.minimum(below, h_sfc) - (below - h), 0.0, None)
    wsum = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, (w * q).sum(axis=0) / np.where(wsum > 0, wsum, 1.0), np.nan)


def _to_faces(tc: np.ndarray, wet: np.ndarray, axis: int, periodic: bool) -> np.ndarray:
    """
    T-point field (ny, nx) → faces along ``axis`` (1: u faces (ny, nx+1); 0: v faces (ny+1, nx)): the mean of the
    wet neighbours, the wet value next to land or a dry cell, 0 between two non-wet cells (MOM6 A-grid stress,
    ``MOM_surface_forcing_gfdl.F90:1086-1097``, with the wet mask).  Periodic: the two edge faces take the wrapped
    pair (last cell, first cell), so they are equal bit for bit.
    """
    t = np.where(wet, tc, 0.0)
    w = wet.astype(float)
    take = (lambda a, s: a[:, s]) if axis == 1 else (lambda a, s: a[s, :])
    if periodic:
        lo_t, lo_w = take(t, slice(-1, None)), take(w, slice(-1, None))
        hi_t, hi_w = take(t, slice(0, 1)), take(w, slice(0, 1))
    else:
        lo_t = lo_w = np.zeros_like(take(t, slice(0, 1)))
        hi_t = hi_w = lo_t
    left_t, left_w = np.concatenate([lo_t, t], axis=axis), np.concatenate([lo_w, w], axis=axis)
    right_t, right_w = np.concatenate([t, hi_t], axis=axis), np.concatenate([w, hi_w], axis=axis)
    n = left_w + right_w
    return np.where(n > 0, (left_t + right_t) / np.where(n > 0, n, 1.0), 0.0)


class SurfaceFluxes:
    """
    The bulk-flux component: ``forcing(st, t)`` → (tau_x, tau_y, fw, heat) for ``Driver.step`` (module docstring),
    and ``last`` with the T-point fields of the last call (``LAST_FIELDS``).  No state.
    """

    def __init__(self, cfg: BulkFluxes, *, fs, area: np.ndarray, ds: np.ndarray, ale: bool, periodic,
                 rho0: float, salinity: bool) -> None:
        self.cfg = cfg
        self.fs, self.ds, self.ale = fs, np.asarray(ds, float), ale
        self.A = np.nan_to_num(np.asarray(area, float))
        self.per_x, self.per_y = bool(getattr(periodic, "x", False)), bool(getattr(periodic, "y", False))
        self.rho0, self.salinity = float(rho0), salinity
        self.h_uv = cfg.h_sfc if cfg.h_uv_sfc is None else cfg.h_uv_sfc
        self.last: dict[str, np.ndarray] = {}

    def _check_time(self, t: float) -> None:
        c0, c1 = self.cfg.atmosphere.coverage()
        if not (c0 <= t <= c1):
            org = np.datetime64(self.cfg.atmosphere.origin, "ns")
            when, a, b = (org + np.timedelta64(int(round(x * 1e9)), "ns") for x in (t, c0, c1))
            raise ValueError(f"bulk surface fluxes: the step needs the atmosphere at {when} (t = {t} s), outside "
                             f"its coverage [{a}, {b}]")

    def forcing(self, st, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        (tau_x (ny, nx+1), tau_y (ny+1, nx) [N m⁻²], fw (ny, nx) [m s⁻¹], heat {"Q_ns", "Q_sw"} [W m⁻²]) from the
        state ``st`` (tⁿ) and the atmosphere at model time ``t`` [s] (the driver passes tⁿ + Δt/2).  Raises
        ValueError naming the date when ``t`` is outside the atmosphere's coverage.
        """
        cfg, fs = self.cfg, self.fs
        self._check_time(float(t))
        atm = cfg.atmosphere(t)
        wet = fs.active & (fs.H + st.eta > fs.d_min) & (self.A > 0)       # the driver's surface-flux mask
        if self.ale:
            h = st.h
        else:
            D = np.maximum(np.nan_to_num(fs.H + st.eta), 0.0)
            h = D[None] * self.ds[:, None, None]
        h = np.where(wet[None], h, 0.0)
        sst = np.where(wet, surface_average(h, st.T, cfg.h_sfc), 10.0)   # land: any finite value, masked below
        sss = np.where(wet, surface_average(h, st.S, cfg.h_sfc), 35.0) if st.S is not None else None
        if cfg.relative_wind:
            uc = 0.5 * (st.u[..., :-1] + st.u[..., 1:])
            vc = 0.5 * (st.v[:, :-1, :] + st.v[:, 1:, :])
            us = np.where(wet, surface_average(h, uc, self.h_uv), 0.0)
            vs = np.where(wet, surface_average(h, vc, self.h_uv), 0.0)
        else:
            us = vs = np.zeros_like(sst)
        u10, v10, t2m, q2m, msl = (atm[k] for k in ("u10", "v10", "t2m", "q2m", "msl"))
        sch = cfg.scheme
        if sch == "coare":
            f = bulk_coare_croco(u10 - us, v10 - vs, t2m - T0_K, q2m, sst, atm["lw_down"], p_sfc=PSURF,
                                 z_u=cfg.z_u, z_t=cfg.z_t, rho0=RHO0_CROCO)
            q_sw, q_lw = (1.0 - cfg.albedo) * atm["sw_down"], f.q_lw          # ruling R2
        else:
            if sch == "ncar":
                f = bulk_ncar_fms(u10, v10, t2m, q2m, sst + T0_K, msl, u_ocean=us, v_ocean=vs,
                                  z_u=cfg.z_u, z_t=cfg.z_t, z_q=cfg.z_q)
            elif sch == "ncar_ly04":
                f = bulk_ncar(u10, v10, t2m, q2m, sst, p=msl, u_ocean=us, v_ocean=vs, z_u=cfg.z_u, z_t=cfg.z_t)
            else:                                                              # 'kara'
                f = bulk_kara(u10, v10, t2m, q2m, sst, p=msl, sss=sss, u_ocean=us, v_ocean=vs)
            q_sw, q_lw = radiative_fluxes(atm["sw_down"], atm["lw_down"], sst, albedo=cfg.albedo,
                                          emissivity=cfg.emissivity)
        z = lambda a: np.where(wet, a, 0.0)                                    # noqa: E731
        # fw in the model's Boussinesq volume convention: mass flux / ρ₀ (module docstring)
        evap = z(f.evap * (EVAP_DENSITY[sch] / self.rho0))
        precip = z(atm["precip"] * (RHO_FW / self.rho0))
        fw = precip - evap
        q_sens, q_lat, q_lw, q_sw = z(f.q_sens), z(f.q_lat), z(q_lw), z(q_sw)
        tx_c, ty_c = z(f.tau_x), z(f.tau_y)
        self.last = dict(q_sens=q_sens, q_lat=q_lat, q_lw=q_lw, q_sw=q_sw, evap=evap, precip=precip, fw=fw,
                         tau_x=tx_c, tau_y=ty_c, sst_bulk=np.where(wet, sst, np.nan))
        tau_x = _to_faces(tx_c, wet, 1, self.per_x)
        tau_y = _to_faces(ty_c, wet, 0, self.per_y)
        heat = {"Q_ns": q_sens + q_lat + q_lw, "Q_sw": q_sw}
        return tau_x, tau_y, fw, heat
