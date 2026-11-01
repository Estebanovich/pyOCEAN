"""
Baroclinic pressure-gradient force of the 3-D core (N1, role PGF): density-Jacobian (Song 1998), two-term
(Haney 1991) or finite-volume (Adcroft et al. 2008, ``core.vertical.pgf.fv_pgf``), with an optional horizontally
uniform reference profile ρ_ref(z) removed first (Berntsen et al. 2015).  Moved unchanged from
``SigmaBaroclinic._baroclinic_pgf`` / ``_fv_pgf`` (same expressions, same order of operations).

With salinity (N2a) every scheme uses ρ(T, S) of the EOS; the finite-volume form then takes the density path
``fv_pgf_rho`` (anomaly ρ − ρ_ref, no division by α), as it does for α = 0 without S.

With a nonlinear EOS (TEOS-10 in 3-D, Task 4; spec 2026-10-31-teos10-3d goal B):

* 'fv' calls ``core.vertical.pgf_nonlinear.fv_pgf_nonlinear`` (MOM6 ``int_density_dz_generic_plm``): without a
  profile (``rho_ref`` None) the density ``rho_pgf_ref`` (MOM6 RHO_PGF_REF, default ρ₀) is subtracted inside the
  EOS; with the T/S reference profile of ``rho_ref="auto"`` (``TSReferenceProfile``) the anomalies T − T_ref(z),
  S − S_ref(z) are reconstructed, so a column equal to the profile has no force;
* 'jacobian' / 'twoterm' use the in-situ density at the Boussinesq pressure of the cell centre,
  ρ(T, S, p(z)) with p = −g ρ₀ z (z from the mean surface, no η; MOM6 convention), minus ρ_ref(z) =
  ρ(T_ref(z), S_ref(z), p(z)) of the profile, or minus ``rho_pgf_ref`` without one.

The linear EOS keeps every expression and its order of operations (bit for bit); there ``mass_weight`` is
"vanished" (the default, today's weighting) or "none" (``fv_pgf(..., mass_weight=False)``).

Reads η [m], T [°C], S [g kg⁻¹] (optional), h [m] (ALE) or the σ-layers; returns the force at u- and v-faces [m s⁻²],
(N, ny, nx+1) and (N, ny+1, nx), k = 0 at the bed.
"""
from __future__ import annotations

import numba
import numpy as np

from core.components.eos import LinearEOS
from core.components.geometry import LayerGeometry
from core.model import threads
from core.model.contract import BuildContext, Issue, Role
from core.model.periodic import NONE
from core.vertical.pgf import fv_pgf, fv_pgf_rho
from core.vertical.pgf_nonlinear import fv_pgf_nonlinear
from physics.eos.convert import pressure_from_z

SCHEMES = ("twoterm", "jacobian", "fv")
MASS_WEIGHTS = ("vanished", "full", "none")      # "full" needs a nonlinear EOS (fv_pgf_nonlinear)


def ts_profile(ref):
    """The T/S reference profile behind ``rho_ref`` (an object with ``z``, ``T``, ``S`` and ``ts(z)``,
    ``core.dynamics.sigma_baroclinic.TSReferenceProfile``), or None for None, "auto" and density profiles."""
    return ref if hasattr(ref, "ts") else None


class PressureGradient:
    """
    PGF component.  ``rho_ref`` is None (ρ₀), "auto" (fixed by ``SigmaBaroclinic.initial_state`` from the
    initial stratification) or a callable z → ρ_ref(z) [kg m⁻³].  ``rho_pgf_ref`` [kg m⁻³]: density subtracted
    without a profile by a nonlinear EOS (None: ρ₀).  ``mass_weight``: bed mass weighting of the 'fv' scheme,
    "vanished" (default), "full" (nonlinear EOS only; NWA12) or "none".
    """
    name = "pgf"
    role = Role.PGF
    reads = ("eta", "T", "S", "h")
    writes = ()
    per = NONE                       # periodic axes (core.model.periodic), set by the model

    def __init__(self, scheme: str, eos: LinearEOS, g: float, rho_ref=None, rho_pgf_ref: float | None = None,
                 mass_weight: str = "vanished") -> None:
        self.scheme, self.eos, self.g, self.rho_ref = scheme, eos, g, rho_ref
        self.rho_pgf_ref = float(eos.rho0 if rho_pgf_ref is None else rho_pgf_ref)
        self.mass_weight = mass_weight

    def check(self, ctx: BuildContext) -> list[Issue]:
        out = []
        if self.scheme not in SCHEMES:
            out.append(Issue("error", self.name, f"unknown pressure-gradient scheme {self.scheme!r}"))
        if self.mass_weight not in MASS_WEIGHTS or (self.eos.linear and self.mass_weight == "full"):
            out.append(Issue("error", self.name, f"pgf_mass_weight={self.mass_weight!r}: use 'vanished' or 'none' "
                                                 "('full' needs a nonlinear EOS)"))
        return out

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}

    def __call__(self, geom: LayerGeometry, eta: np.ndarray, T: np.ndarray,
                 h: np.ndarray | None = None, S: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Baroclinic pressure-gradient force at u- and v-faces [m s⁻²].  ``S`` [g kg⁻¹]: salinity (N2a), None
        without; ρ = ρ(T, S) of the linear EOS (Vallis 2006 eq. 1.156).

        ρ′ = ρ − ρ_ref(z) with ρ_ref = ρ₀, or a reference profile ``rho_ref(z)``
        (Haney 1991; the "balanced" method of Berntsen et al. 2015): the pressure of
        a horizontally uniform ρ_ref(z) has no gradient at fixed z, so subtracting it
        removes the large cancelling terms of a resting stratification.

        'twoterm':  PGF = −(1/ρ₀)[∂P′/∂x|_σ + g ρ′ ∂z/∂x|_σ]   (second order; large errors
                     over steep slopes — Haney 1991, seamount max|u| ~50 cm/s)
        'jacobian': density-Jacobian form (Song 1998), integrated down from the surface
                     between layer centres k and k+1:
            ∂P′/∂x|_z(z_k) = ∂P′/∂x|_z(z_{k+1}) + (g/Δx)[δ_x ρ̄ · (δ_k z)‾ − δ_x z̄ · (δ_k ρ)‾]
                     starting at the surface z = η with ρ′ extrapolated linearly from the two
                     top layers.  Exact for ρ linear in z over any topography.
        """
        if self.scheme == "fv":
            return self._fv(geom, eta, T, h, S)
        n = T.shape[0]
        if h is None:
            D = np.maximum(np.nan_to_num(geom.H + eta, nan=geom.d_min), geom.d_min)
            h = D[None] * geom.ds[:, None, None]
            z = -np.nan_to_num(geom.H)[None] + D[None] * geom.sigma_c[:, None, None]   # cell-centre height (z = 0 at rest)
        else:
            z = geom.layer_heights(eta, h)
        if self.eos.linear:
            rho = self.eos.density(T, S)
            rp = rho - (self.rho_ref(z) if self.rho_ref is not None else self.eos.rho0)
        else:                                                         # in situ at p = −g ρ₀ z (module docstring)
            rho = self.eos.density(T, S, pressure_from_z(z, self.eos.rho0, self.g))
            ref = None if isinstance(self.rho_ref, str) else self.rho_ref        # "auto" before initial_state
            rp = rho - (ref(z) if ref is not None else self.rho_pgf_ref)
        g = self.g
        # surface values for the Jacobian: z_s = η, ρ′ linearly extrapolated from the two top layers
        z_s = np.nan_to_num(eta)[None]
        if n > 1:
            r_s = (rp[-1] + (rp[-1] - rp[-2]) * (z_s[0] - z[-1]) / np.where(z[-1] - z[-2] != 0, z[-1] - z[-2], 1.0))[None]
        else:
            r_s = rp[-1:]
        if self.scheme == "twoterm":
            above = np.cumsum((rp * h)[::-1], axis=0)[::-1] - rp * h       # ∫ over the layers above
            P = g * (above + 0.5 * rp * h)

        def along(a_lo, a_hi, z_lo, z_hi, P_lo=None, P_hi=None, s_lo=None, s_hi=None):
            """∂P′/∂x|_z × Δx at faces between the 'lo' and 'hi' columns, all layers."""
            if self.scheme == "twoterm":
                return (P_hi - P_lo) + g * 0.5 * (a_hi + a_lo) * (z_hi - z_lo)
            # Jacobian from the surface (z = η, P′ = 0 there; surface term g ρ̄′_s δ_x η) down to the
            # top layer centre, then between consecutive layer centres
            (rs_lo, zs_lo), (rs_hi, zs_hi) = s_lo, s_hi
            a_lo, a_hi = np.concatenate([a_lo, rs_lo]), np.concatenate([a_hi, rs_hi])
            z_lo, z_hi = np.concatenate([z_lo, zs_lo]), np.concatenate([z_hi, zs_hi])
            out = np.empty(a_lo.shape)
            out[-1] = g * 0.5 * (rs_hi[0] + rs_lo[0]) * (zs_hi[0] - zs_lo[0])
            rbar_lo, rbar_hi = 0.5 * (a_lo[1:] + a_lo[:-1]), 0.5 * (a_hi[1:] + a_hi[:-1])
            zbar_lo, zbar_hi = 0.5 * (z_lo[1:] + z_lo[:-1]), 0.5 * (z_hi[1:] + z_hi[:-1])
            dz = 0.5 * ((z_lo[1:] - z_lo[:-1]) + (z_hi[1:] - z_hi[:-1]))
            dr = 0.5 * ((a_lo[1:] - a_lo[:-1]) + (a_hi[1:] - a_hi[:-1]))
            inc = g * ((rbar_hi - rbar_lo) * dz - (zbar_hi - zbar_lo) * dr)   # pair (k, k+1), k = 0 … N−1 (N = surface)
            out[:-1] = out[-1] + np.cumsum(inc[::-1], axis=0)[::-1]
            return out[:-1]

        gx = np.zeros((n,) + geom.H_u.shape)
        gy = np.zeros((n,) + geom.H_v.shape)
        if self.per:
            return self._wrapped(along, gx, gy, geom, rp, z, r_s, z_s, P if self.scheme == "twoterm" else None)
        okx = np.isfinite(geom.H_u[:, 1:-1])
        oky = np.isfinite(geom.H_v[1:-1, :])
        if self.scheme == "twoterm":
            ax = along(rp[..., :-1], rp[..., 1:], z[..., :-1], z[..., 1:], P[..., :-1], P[..., 1:])
            ay = along(rp[:, :-1], rp[:, 1:], z[:, :-1], z[:, 1:], P[:, :-1], P[:, 1:])
        else:
            ax = along(rp[..., :-1], rp[..., 1:], z[..., :-1], z[..., 1:],
                       s_lo=(r_s[..., :-1], z_s[..., :-1]), s_hi=(r_s[..., 1:], z_s[..., 1:]))
            ay = along(rp[:, :-1], rp[:, 1:], z[:, :-1], z[:, 1:],
                       s_lo=(r_s[:, :-1], z_s[:, :-1]), s_hi=(r_s[:, 1:], z_s[:, 1:]))
        gx[:, :, 1:-1] = np.where(okx, -ax / geom.du[:, 1:-1] / self.eos.rho0, 0.0)
        gy[:, 1:-1, :] = np.where(oky, -ay / geom.dv[1:-1, :] / self.eos.rho0, 0.0)
        return gx, gy

    def _wrapped(self, along, gx, gy, geom: LayerGeometry, rp, z, r_s, z_s, P):
        """
        The 'jacobian' / 'twoterm' force with periodic axes (N3 task 3): the column pairs along a periodic axis are
        (n−1, 0), (0, 1), …, (n−2, n−1) — the cell arrays extended by the last cell in front — so faces 0 … n−1 are
        computed in the usual way and face n is face 0.  Closed axes as without periodicity.
        """
        rho0 = self.eos.rho0
        for axis, g, Hf, d, on in ((-1, gx, geom.H_u, geom.du, self.per.x), (-2, gy, geom.H_v, geom.dv, self.per.y)):
            ext = (lambda a: np.concatenate([np.take(a, [-1], axis=axis), a], axis=axis)) if on else (lambda a: a)
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            a_, z_ = ext(rp), ext(z)
            if P is not None:
                a = along(a_[lo], a_[hi], z_[lo], z_[hi], ext(P)[lo], ext(P)[hi])
            else:
                rs_, zs_ = ext(r_s), ext(z_s)
                a = along(a_[lo], a_[hi], z_[lo], z_[hi], s_lo=(rs_[lo], zs_[lo]), s_hi=(rs_[hi], zs_[hi]))
            n_f = Hf.shape[axis]                                   # faces n + 1
            fs = slice(0, n_f - 1) if on else slice(1, n_f - 1)  # faces the pairs above give
            sl2 = [slice(None)] * 2
            sl2[axis] = fs
            ok = np.isfinite(Hf[tuple(sl2)])
            sl3 = [slice(None)] * 3
            sl3[axis] = fs
            g[tuple(sl3)] = np.where(ok, -a / d[tuple(sl2)] / rho0, 0.0)
            if on:
                last, first = [slice(None)] * 3, [slice(None)] * 3
                last[axis], first[axis] = n_f - 1, 0
                g[tuple(last)] = g[tuple(first)]
        return gx, gy

    def _fv(self, geom: LayerGeometry, eta: np.ndarray, T: np.ndarray,
            h: np.ndarray | None, S: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Finite-volume PGF (core.vertical.pgf) on σ-layers (h=None) or ALE layers, with ρ_ref(z) if set.  T only
        with α ≠ 0: ``fv_pgf`` (the T′ anomaly, unchanged arithmetic).  With S, or α = 0 (where T′ = T − T₀ −
        (ρ₀ − ρ_ref)/α is 0/0): ``fv_pgf_rho`` on ρ(T, S) − ρ_ref(z), which never divides by α.
        """
        if h is None:
            D = np.maximum(np.nan_to_num(geom.H + eta, nan=geom.d_min), geom.d_min)
            h = D[None] * geom.ds[:, None, None]
        if not self.eos.linear:
            return self._fv_nonlinear(geom, eta, T, h, S)
        ref = None if isinstance(self.rho_ref, str) else self.rho_ref        # "auto" before initial_state: ρ₀
        numba.set_num_threads(threads._THREADS[0])
        kinds = ("c", "c", "c", "c", "u", "v")                            # periodic: pad–crop, halo 1 (two columns)
        kw = {} if self.mass_weight == "vanished" else {"mass_weight": False}   # "none": no bed mass weighting
        if S is None and self.eos.t_coef != 0:
            args = (h, np.nan_to_num(geom.H), np.nan_to_num(eta), np.nan_to_num(T), geom.du, geom.dv)
            fn = lambda *a: fv_pgf(*a, self.eos.rho0, self.eos.t_coef, self.eos.t_ref, ref, self.g, **kw)  # noqa: E731
        else:
            rho = self.eos.density(np.nan_to_num(T), None if S is None else np.nan_to_num(S))
            args = (h, np.nan_to_num(geom.H), np.nan_to_num(eta), rho, geom.du, geom.dv)
            fn = lambda *a: fv_pgf_rho(*a, self.eos.rho0, ref, self.g, **kw)                       # noqa: E731
        gx, gy = self.per.apply(fn, args, kinds, ("u", "v"), 1) if self.per else fn(*args)
        return (np.where(np.isfinite(geom.H_u)[None], gx, 0.0),
                np.where(np.isfinite(geom.H_v)[None], gy, 0.0))

    def _fv_nonlinear(self, geom: LayerGeometry, eta: np.ndarray, T: np.ndarray, h: np.ndarray,
                      S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Finite-volume PGF with a nonlinear EOS (``fv_pgf_nonlinear``, MOM6 ``int_density_dz_generic_plm``,
        MOM_density_integrals.F90): the T/S profile mode with ``rho_ref="auto"``, MOM6's mode with
        ``rho_pgf_ref`` otherwise.  Active columns are the wet ones (D = H + η > d_min); the kernel takes the
        surface pressure from η and the interfaces from Σh − H, so η is passed as Σh − H in wet columns (exactly
        the height of the column top that h describes, also in ALE).
        """
        H0 = np.nan_to_num(geom.H)
        hh = np.nan_to_num(h)
        wet = np.isfinite(geom.H) & (np.nan_to_num(geom.H + eta, nan=0.0) > geom.d_min)
        eta_h = np.where(wet, hh.sum(axis=0) - H0, np.nan_to_num(eta))
        prof = ts_profile(self.rho_ref)
        ref = {} if prof is None else {"T_ref": prof.T, "S_ref": prof.S, "z_ref": prof.z}
        numba.set_num_threads(threads._THREADS[0])

        def fn(h_, H_, e_, T_, S_, du, dv, wet_):
            return fv_pgf_nonlinear(h_, H_, e_, T_, S_, du, dv, eos_name=self.eos.name, rho0=self.eos.rho0,
                                    rho_pgf_ref=self.rho_pgf_ref, g=self.g, mass_weight=self.mass_weight,
                                    wet=wet_, **ref)
        args = (hh, H0, eta_h, np.nan_to_num(T), np.nan_to_num(S), geom.du, geom.dv, wet)
        kinds = ("c", "c", "c", "c", "c", "u", "v", "c")
        gx, gy = self.per.apply(fn, args, kinds, ("u", "v"), 1) if self.per else fn(*args)
        return (np.where(np.isfinite(geom.H_u)[None], gx, 0.0),
                np.where(np.isfinite(geom.H_v)[None], gy, 0.0))
