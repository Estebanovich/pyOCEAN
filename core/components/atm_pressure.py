"""
Atmospheric pressure forcing of the 3-D core (regional chain Task 6; reference read
docs/design/regional_refs/mom6_psurf_runoff.md §2).

**Interior.**  In a hydrostatic Boussinesq model the sea-level pressure p_a(x, y, t) adds p_a to the pressure of
every depth, so the momentum equations gain the depth-uniform acceleration

    ∂u/∂t = … − g ∇η − (1/ρ₀) ∇p_a                                        (Wunsch & Stammer 1997, Rev. Geophys. 35, eq. 1)  [verificar eq.]

MOM6 puts p_a at the top interface of its finite-volume PGF, pa(1) = ρ_ref g (e(1) − Z_ref) + p_atm
(``src/core/MOM_PressureForce_FV.F90:1250-1260``, MOM6 git 0e9a561), so every layer feels −∇p_a/ρ₀, and its
barotropic solver receives that force inside the layer accelerations together with an ``eta_PF`` consistent with it
(``MOM_barotropic.F90:3304-3313``): MOM6 treats p_a as a surface-height anomaly p_a/(ρ₀ g) in both modes.  pyOCEAN
adds F = −(1/ρ₀)∇p_a to the momentum tendency of every layer after the AB3 combination (time-centred at
t^{n+½}, not extrapolated, as the wind stress and the body tide of Task 5).  Being depth-uniform it enters only the
depth mean ū, i.e. the same θ-implicit free-surface solve where g∇η enters (``WetDryFreeSurface.step``); the
baroclinic u′ is untouched.  The gradient is the free surface's own operator (``WetDryFreeSurface._gradient``:
differences over the u-face distance ``dx_u`` and the v-face distance ``dy_v``, the periodic wrap face from its
two cells), so on any grid, curvilinear included (Task 3 metric audit, docs/design/regional_refs/metric_audit.md),
the static inverse barometer

    η_IB = −(p_a − p̄_a) / (ρ₀ g)                                           (inverse barometer)

balances −g∇η − ∇p_a/ρ₀ = 0 exactly, face by face.  With the scalar self-attraction and loading of Task 5
(``sal_beta`` = β) the surface gradient is g(1 − β)∇η, so the discrete equilibrium is −(p_a − p̄_a)/(ρ₀ g (1 − β));
η_IB here uses the same g(1 − β) (``fs.g_pgf``) so that the edge exterior below matches the interior equilibrium
(class c; the observed inverse barometer is −p′/(ρ₀ g), and NWA12 applies no p_a at all, see below).

**Not in the EOS pressure.**  The density of the nonlinear equations of state is evaluated at p = −g ρ₀ z without
η and without p_a, as MOM6 (docs/design/teos10_refs/mom6_eos.md; ``SSH_IN_EOS_PRESSURE_FOR_PGF`` False in NWA12,
``MOM_PressureForce_FV.F90:1262-1275``).  A 5 kPa change of p_a changes ρ by ~0.002 kg m⁻³ (compressibility
≈ 4.4e-7 kg m⁻³ Pa⁻¹), negligible next to the error of omitting η in the same pressure.

**Open boundaries.**  At every ``'radiating'`` edge the Flather exterior elevation gets the local inverse
barometer, η_ext ← η_ext + η_IB at the edge faces (η_IB of the boundary cell), so the exterior is in the
equilibrium the interior force drives toward and a static or slow p_a does not make the Flather condition push
water in or out.  p̄_a is the area-weighted mean of p_a over the wet T cells (active, H + ηⁿ > d_min, area > 0: the
bulk-flux mask of the state n) at the same time.  Taking the domain mean means a spatially uniform p_a has no
effect (the regional domain cannot exchange mass with the rest of the ocean in response to it); ``p_ref`` (Pa) fixes
p̄_a instead (SIS2 subtracts 1e5 Pa when APPLY_SLP_TO_OCEAN, ``SIS2/src/ice_model.F90:779-784``), so a uniform
change of p_a moves the whole domain through the open edges.  'clamped' and 'eta' edges are not changed (their
η is the provider's).

**Time level.**  p_a is the ``msl`` of the bulk component's atmosphere call (``SurfaceFluxes.last_msl``), at
t^{n+½} = tⁿ + Δt/2; both the interior force and the edge η_IB use that one field.  The time-varying exterior and
the boundary tide are at tⁿ⁺¹ (Task 5, MOM6 ``update_OBC_data`` at Time_local); using t^{n+½} for η_IB as well keeps
the edge and the interior in the same balance at every step, at the price of a lag Δt/2 ∂η_IB/∂t at the edge
(≈ 3e-6 m for a 5 hPa/day change at Δt = 600 s; class c).  No atmosphere evaluation is added.

**NWA12.**  The MOM6 counterpart has no atmospheric pressure: SIS2 ``APPLY_SLP_TO_OCEAN = False`` and
``MAX_P_SURF = 0`` (mom6_psurf_runoff.md §1), so ``atm_pressure=True`` is a pyOCEAN option to be compared with
tide-gauge observations, not a parity setting.

No state: the field is a pure function of time through the atmosphere provider; a restart checks the flag and
``p_ref`` only.  Index order (j, i) at T points; u-faces (ny, nx+1), v-faces (ny+1, nx).
"""
from __future__ import annotations

import numpy as np

#: Edge → (face kind, slice of the edge faces, slice of their boundary cells), as ``boundary_tides.EDGE_FACES``.
_EDGE = {"west": ("u", np.s_[:, 0], np.s_[:, 0]), "east": ("u", np.s_[:, -1], np.s_[:, -1]),
         "south": ("v", np.s_[0, :], np.s_[0, :]), "north": ("v", np.s_[-1, :], np.s_[-1, :])}


class AtmosphericPressure:
    """
    Atmospheric pressure component (module docstring): the force −∇p_a/ρ₀ at the faces and the inverse barometer
    η_IB of the 'radiating' edges.

    Parameters
    ----------
    fs : core.dynamics.wetting_drying.WetDryFreeSurface
        The model's free surface (its gradient operator, active faces, depths, d_min and g(1 − β)).
    area : (ny, nx) cell areas [m²] (NaN or 0 on land).
    rho0 : Boussinesq reference density ρ₀ [kg m⁻³].
    edges : the 'radiating' open edges that get η_IB on their Flather exterior.
    p_ref : fixed p̄_a [Pa]; None (default): the area mean over the wet T cells at each step.
    """

    def __init__(self, fs, area: np.ndarray, rho0: float, edges: tuple[str, ...] = (),
                 p_ref: float | None = None) -> None:
        self.fs = fs
        self.A = np.nan_to_num(np.asarray(area, float))
        self.rho0 = float(rho0)
        self.edges = tuple(edges)
        self.p_ref = None if p_ref is None else float(p_ref)
        self._act_u, self._act_v = np.isfinite(fs.H_u), np.isfinite(fs.H_v)

    def wet(self, eta: np.ndarray) -> np.ndarray:
        """Wet T cells of the state with surface ``eta`` [m] (the bulk-flux mask of the driver)."""
        fs = self.fs
        return fs.active & (fs.H + eta > fs.d_min) & (self.A > 0)

    def mean(self, p_a: np.ndarray, wet: np.ndarray) -> float:
        """p̄_a [Pa]: ``p_ref``, or Σ A p_a / Σ A over the wet cells (p_a itself if none is wet)."""
        if self.p_ref is not None:
            return self.p_ref
        w = np.where(wet, self.A, 0.0)
        s = w.sum()
        return float((w * np.where(wet, p_a, 0.0)).sum() / s) if s > 0 else float(np.nanmean(p_a))

    def force(self, p_a: np.ndarray, pbar: float) -> tuple[np.ndarray, np.ndarray]:
        """
        (F_x, F_y) = −(1/ρ₀) ∇p_a at u- and v-faces [m s⁻²]: the free surface's gradient of the anomaly
        p_a − p̄_a (the mean drops out of the gradient; subtracting it first keeps ~5 digits that a 1e5 Pa field
        would lose), zero on inactive faces and on domain-edge faces (as ∇η).
        """
        gx, gy = self.fs._gradient(np.asarray(p_a, float) - pbar)
        return np.where(self._act_u, -gx / self.rho0, 0.0), np.where(self._act_v, -gy / self.rho0, 0.0)

    def eta_ib(self, p_a: np.ndarray, pbar: float) -> np.ndarray:
        """η_IB = −(p_a − p̄_a)/(ρ₀ g(1 − β)) [m] at the T points (module docstring)."""
        return -(np.asarray(p_a, float) - pbar) / (self.rho0 * self.fs.g_pgf)

    def flather_eta(self, eta_ib: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """(η_u, η_v) of the Flather exterior: η_IB of the boundary cell at the faces of each 'radiating' edge,
        0 elsewhere; None without such an edge."""
        if not self.edges:
            return None
        eu, ev = np.zeros(self.fs.H_u.shape), np.zeros(self.fs.H_v.shape)
        for e in self.edges:
            kind, sl_f, sl_c = _EDGE[e]
            (eu if kind == "u" else ev)[sl_f] = np.where(self.fs.active[sl_c], eta_ib[sl_c], 0.0)
        return eu, ev

    def config(self) -> dict[str, str | float]:
        """Values a restart must match (no state)."""
        return dict(atm_pressure="on", atm_pressure_ref="none" if self.p_ref is None else self.p_ref,
                    atm_pressure_edges=",".join(self.edges) or "none")
