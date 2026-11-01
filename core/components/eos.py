"""
Equation of state of the 3-D core (N1, role EOS).

Linear in temperature and salinity (N2a; TEOS-10 enters later in N2):

    ρ = ρ₀ − α (T − T₀) + β (S − S₀)   [kg m⁻³]   (Vallis 2006, eq. 1.156)

ρ₀ [kg m⁻³] reference density, α = ``t_coef`` [kg m⁻³ °C⁻¹], T₀ = ``t_ref`` [°C], β = ``s_coef``
[kg m⁻³ (g kg⁻¹)⁻¹], S₀ = ``s_ref`` [g kg⁻¹].  Without S (``density(T)``) the expression is exactly the
temperature-only one of phase 2, so runs without salinity stay bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from core.model.contract import BuildContext, Issue, Role, param
from physics.eos import roquet_rho, wright_full
from physics.eos.convert import freezing_linear, freezing_teos10, require_gsw


@runtime_checkable
class EOS(Protocol):
    """
    What the 3-D core needs of an equation of state.  ``ts_kind`` is the (T, S) convention the EOS
    takes: ``"theta_sp"`` (θ [°C], S_P [PSU]) or ``"ct_sa"`` (Θ [°C], S_A [g kg⁻¹]); p is the sea
    pressure [Pa].  ``linear`` marks the EOS whose ρ does not depend on p.
    """

    name: str
    rho0: float
    linear: bool
    ts_kind: str
    cp_default: float

    def density(self, T, S=None, p=None): ...
    def density_anomaly(self, T, S, p, rho_ref): ...
    def density_derivs(self, T, S, p): ...
    def potential_density(self, T, S, p_ref): ...
    def freezing_point(self, S, p): ...


@dataclass(frozen=True)
class LinearEOS:
    rho0: float = param(1000.0, units="kg m-3", doc="reference density ρ₀", cite="Vallis 2006 eq. 1.156")
    t_coef: float = param(0.2, units="kg m-3 degC-1", doc="thermal coefficient α (ρ₀ times the expansion coefficient)")
    t_ref: float = param(0.0, units="degC", doc="reference temperature T₀")
    s_coef: float = param(0.0, units="kg m-3 (g kg-1)-1", doc="haline coefficient β (ρ₀ times the contraction "
                          "coefficient)", cite="Vallis 2006 eq. 1.156")
    s_ref: float = param(35.0, units="g kg-1", doc="reference salinity S₀")

    name = "linear"
    linear = True
    ts_kind = "theta_sp"
    cp_default = 3985.0  # J kg-1 degC-1 (MITgcm/CROCO-like value)
    role = Role.EOS
    reads = ("T", "S")
    writes = ()

    def density(self, T: np.ndarray, S: np.ndarray | None = None, p: np.ndarray | None = None) -> np.ndarray:
        """
        Linear EOS (p is accepted and ignored) [kg m⁻³] (Vallis 2006, eq. 1.156): ρ = ρ₀ − α (T − T₀) without S (exactly the phase-2
        expression), ρ = ρ₀ − α (T − T₀) + β (S − S₀) with S [g kg⁻¹] (same shape as T).
        """
        if S is None:
            return self.rho0 - self.t_coef * (T - self.t_ref)
        return self.rho0 - self.t_coef * (T - self.t_ref) + self.s_coef * (S - self.s_ref)

    def density_anomaly(self, T, S, p, rho_ref: float) -> np.ndarray:
        """ρ − ρ_ref [kg m⁻³] (p ignored)."""
        return self.density(T, S) - rho_ref

    def density_derivs(self, T, S=None, p=None) -> tuple[np.ndarray, np.ndarray]:
        """(∂ρ/∂T, ∂ρ/∂S) = (−α, β) [kg m⁻³ °C⁻¹, kg m⁻³ (g kg⁻¹)⁻¹], broadcast to the shape of T."""
        shape = np.shape(T)
        return np.full(shape, -self.t_coef), np.full(shape, self.s_coef)

    def potential_density(self, T, S, p_ref=None) -> np.ndarray:
        """Density at the reference pressure; for the linear EOS the in-situ density."""
        return self.density(T, S)

    def freezing_point(self, S, p):
        """MOM6 linear freezing point [°C] (θ, S_P)."""
        return freezing_linear(S, p)

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}


class _NonlinearEOS:
    """Shared protocol methods of the pressure-dependent EOS (a module of Task 1 does the work)."""

    _mod = None
    role = Role.EOS
    reads = ("T", "S")
    writes = ()
    linear = False

    def density(self, T, S=None, p=None) -> np.ndarray:
        """ρ(T, S, p) [kg m⁻³]; p [Pa] defaults to the surface (0)."""
        if S is None:
            raise ValueError(f"{self.name} EOS needs salinity")
        return self._mod.density(T, S, 0.0 if p is None else p)

    def density_anomaly(self, T, S, p, rho_ref: float) -> np.ndarray:
        """ρ − ρ_ref [kg m⁻³] without the cancellation of the subtraction (MOM6 form)."""
        return self._mod.density_anomaly(T, S, p, rho_ref)

    def density_derivs(self, T, S, p) -> tuple[np.ndarray, np.ndarray]:
        """(∂ρ/∂T, ∂ρ/∂S) at constant p."""
        return self._mod.density_derivs(T, S, p)

    def potential_density(self, T, S, p_ref) -> np.ndarray:
        """Density referenced to p_ref [Pa] (e.g. σ₂: p_ref = 2e7)."""
        return self._mod.density(T, S, p_ref)

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}


@dataclass(frozen=True)
class WrightFullEOS(_NonlinearEOS):
    """Wright (1997) full-range EOS on (θ, S_P) (MOM6 WRIGHT_FULL)."""

    rho0: float = param(1035.0, units="kg m-3", doc="Boussinesq reference density ρ₀")

    name = "wright_full"
    ts_kind = "theta_sp"
    cp_default = 3991.86795711963  # J kg-1 K-1, MOM6 default C_P
    _mod = wright_full

    def freezing_point(self, S, p):
        """MOM6 linear freezing point [°C] (θ, S_P)."""
        return freezing_linear(S, p)


@dataclass(frozen=True)
class TEOS10EOS(_NonlinearEOS):
    """Roquet et al. (2015) TEOS-10 polynomial EOS on (Θ, S_A) (MOM6 ROQUET_RHO)."""

    rho0: float = param(1035.0, units="kg m-3", doc="Boussinesq reference density ρ₀")

    name = "teos10"
    ts_kind = "ct_sa"
    cp_default = 3991.86795711963  # J kg-1 K-1, TEOS-10 c_p0
    _mod = roquet_rho

    def freezing_point(self, S, p):
        """TEOS-10 freezing Θ_f [°C] of S_A at p [Pa]."""
        return freezing_teos10(S, p)


def make_eos(name: str, rho0: float | None = None, **linear_params):
    """
    EOS component by name: ``"linear"`` (takes ``linear_params``: t_coef, t_ref, s_coef, s_ref),
    ``"wright_full"`` or ``"teos10"``.  ``rho0`` None keeps the class default.  ``"teos10"`` needs the optional
    ``gsw`` package (Θ ↔ θ conversions at the I/O boundary and of the bulk SST, freezing point on cold cells) and
    raises ImportError here, at construction, when it is missing.
    """
    kw = {} if rho0 is None else {"rho0": rho0}
    if name == "linear":
        return LinearEOS(**kw, **linear_params)
    if linear_params:
        raise ValueError(f"linear_params {sorted(linear_params)} only apply to the 'linear' EOS")
    if name == "wright_full":
        return WrightFullEOS(**kw)
    if name == "teos10":
        require_gsw("eos='teos10' (Θ/S_A conversions and freezing point)")
        return TEOS10EOS(**kw)
    raise ValueError(f"unknown EOS {name!r}: expected 'linear', 'wright_full' or 'teos10'")
