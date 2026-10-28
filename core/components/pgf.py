"""
Baroclinic pressure-gradient force of the 3-D core (N1, role PGF): density-Jacobian (Song 1998), two-term
(Haney 1991) or finite-volume (Adcroft et al. 2008, ``core.vertical.pgf.fv_pgf``), with an optional horizontally
uniform reference profile ρ_ref(z) removed first (Berntsen et al. 2015).  Moved unchanged from
``SigmaBaroclinic._baroclinic_pgf`` / ``_fv_pgf`` (same expressions, same order of operations).

With salinity (N2a) every scheme uses ρ(T, S) of the EOS; the finite-volume form then takes the density path
``fv_pgf_rho`` (anomaly ρ − ρ_ref, no division by α), as it does for α = 0 without S.

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
from core.vertical.pgf import fv_pgf, fv_pgf_rho

SCHEMES = ("twoterm", "jacobian", "fv")


class PressureGradient:
    """
    PGF component.  ``rho_ref`` is None (ρ₀), "auto" (fixed by ``SigmaBaroclinic.initial_state`` from the
    initial stratification) or a callable z → ρ_ref(z) [kg m⁻³].
    """
    name = "pgf"
    role = Role.PGF
    reads = ("eta", "T", "S", "h")
    writes = ()

    def __init__(self, scheme: str, eos: LinearEOS, g: float, rho_ref=None) -> None:
        self.scheme, self.eos, self.g, self.rho_ref = scheme, eos, g, rho_ref

    def check(self, ctx: BuildContext) -> list[Issue]:
        if self.scheme not in SCHEMES:
            return [Issue("error", self.name, f"unknown pressure-gradient scheme {self.scheme!r}")]
        return []

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
        rho = self.eos.density(T, S)
        rp = rho - (self.rho_ref(z) if self.rho_ref is not None else self.eos.rho0)
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
        ref = None if isinstance(self.rho_ref, str) else self.rho_ref        # "auto" before initial_state: ρ₀
        numba.set_num_threads(threads._THREADS[0])
        if S is None and self.eos.t_coef != 0:
            gx, gy = fv_pgf(h, np.nan_to_num(geom.H), np.nan_to_num(eta), np.nan_to_num(T), geom.du, geom.dv,
                            self.eos.rho0, self.eos.t_coef, self.eos.t_ref, ref, self.g)
        else:
            rho = self.eos.density(np.nan_to_num(T), None if S is None else np.nan_to_num(S))
            gx, gy = fv_pgf_rho(h, np.nan_to_num(geom.H), np.nan_to_num(eta), rho, geom.du, geom.dv,
                                self.eos.rho0, ref, self.g)
        return (np.where(np.isfinite(geom.H_u)[None], gx, 0.0),
                np.where(np.isfinite(geom.H_v)[None], gy, 0.0))
