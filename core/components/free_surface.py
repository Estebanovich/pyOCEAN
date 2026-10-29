"""
Free surface plus vertical viscosity of the 3-D core (N1, role FREE_SURFACE).

In N1 this component wraps ``core.dynamics.sigma_wd.SigmaWetDry.step`` unchanged: the implicit vertical viscosity
with bed drag and the wind as a top flux, the θ-implicit free surface (Casulli & Cheng 1992; MITgcm's θ-method)
with Flather faces and the barotropic correction.  Since N3 task 1 the turbulence closure is a component of its own
(``core.components.closure``): the driver hands its ν at the face interfaces (``nu_u``, ``nu_v``) to this step and
updates the closure afterwards (``SigmaWetDry.closure_forcing``), in CROCO's order.  Splitting viscosity, drag and
free surface further is planned (spec 2026-10-28 §2).

What the free surface leaves behind for the rest of the step (the faces that carried flow, the upwind face depths
used in the fluxes, the Flather fluxes and depths at the open faces) is **returned** as ``BarotropicFluxes``
instead of being read later from ``fs.last_*``.  The lagged-LU preconditioner of the CG solve stays in the free
surface (a cache that affects the bits; it is written to the restart, ``state_fields``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.model.contract import BuildContext, Issue, Role


@dataclass(frozen=True)
class BarotropicFluxes:
    """Free-surface step products read by the layer transports and the open boundaries (u-faces (ny, nx+1),
    v-faces (ny+1, nx)); ``open_*`` None without Flather faces."""
    wet_u: np.ndarray            # faces that carried flow in this step
    wet_v: np.ndarray
    Du: np.ndarray               # upwind total face depth used in the fluxes [m]
    Dv: np.ndarray
    open_Qu: np.ndarray | None   # θ-weighted Flather volume flux [m³ s⁻¹] (model sign)
    open_Qv: np.ndarray | None
    open_Du: np.ndarray | None   # total depth at the Flather faces [m]
    open_Dv: np.ndarray | None
    clip_volume: float           # volume added by the positivity guard [m³] (diagnostic)
    fw: np.ndarray | None = None  # freshwater source used by the free surface [m s⁻¹] (0 on land / dry), N2a

    @classmethod
    def from_free_surface(cls, fs) -> "BarotropicFluxes":
        has_open = bool(fs._flather)
        get = (lambda name: getattr(fs, name)) if has_open else (lambda name: None)   # noqa: E731
        return cls(fs.last_wet_u, fs.last_wet_v, fs.last_Du, fs.last_Dv, get("last_open_Qu"), get("last_open_Qv"),
                   get("last_open_Du"), get("last_open_Dv"), fs.last_clip_volume, getattr(fs, "last_fw", None))


class FreeSurfaceVerticalViscosity:
    name = "free_surface"
    role = Role.FREE_SURFACE
    reads = ("eta", "u", "v")
    writes = ("eta", "u", "v")

    def __init__(self, core) -> None:
        self.core = core                     # SigmaWetDry
        self.fs = core.fs

    def step(self, eta: np.ndarray, u: np.ndarray, v: np.ndarray, gu_ab: np.ndarray, gv_ab: np.ndarray, tu, tv,
             flather: dict, fw: np.ndarray | None = None, nu_u: np.ndarray | None = None,
             nu_v: np.ndarray | None = None, **visc) -> tuple[np.ndarray, np.ndarray, np.ndarray, BarotropicFluxes]:
        """
        One free-surface + vertical-viscosity step: (ηⁿ⁺¹, uⁿ⁺¹, vⁿ⁺¹) before the open-boundary relaxation and
        the barotropic fluxes of the step.  ``nu_u``, ``nu_v``: ν at the face interfaces from the closure
        component (``Closure.viscosity``), None: the solver's constant ν.  ``visc``: the ALE face thicknesses, face
        fractions (wu, wv) and bed-drag rates of ``SigmaWetDry.step``.  ``fw``: surface freshwater flux P − E
        [m s⁻¹], a real volume source of the free surface (N2a; masked to wet cells, returned as
        ``BarotropicFluxes.fw``); None adds no operation.
        """
        eta_new, u_new, v_new = self.core.step(eta, u, v, G_u=gu_ab, G_v=gv_ab, tau_u=tu, tau_v=tv, **visc, **flather,
                                               fw=fw, nu_u=nu_u, nu_v=nu_v)
        return eta_new, u_new, v_new, BarotropicFluxes.from_free_surface(self.fs)

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        pre = self.fs.preconditioner_state() if hasattr(self.fs, "preconditioner_state") else None
        return {} if pre is None else {"fs_lu_data": pre["data"], "fs_lu_indices": pre["indices"],
                                       "fs_lu_indptr": pre["indptr"]}
