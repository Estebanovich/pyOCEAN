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

import numpy as np

from core.model.contract import BuildContext, Issue, Role, param


@dataclass(frozen=True)
class LinearEOS:
    rho0: float = param(1000.0, units="kg m-3", doc="reference density ρ₀", cite="Vallis 2006 eq. 1.156")
    t_coef: float = param(0.2, units="kg m-3 degC-1", doc="thermal coefficient α (ρ₀ times the expansion coefficient)")
    t_ref: float = param(0.0, units="degC", doc="reference temperature T₀")
    s_coef: float = param(0.0, units="kg m-3 (g kg-1)-1", doc="haline coefficient β (ρ₀ times the contraction "
                          "coefficient)", cite="Vallis 2006 eq. 1.156")
    s_ref: float = param(35.0, units="g kg-1", doc="reference salinity S₀")

    name = "linear_eos"
    role = Role.EOS
    reads = ("T", "S")
    writes = ()

    def density(self, T: np.ndarray, S: np.ndarray | None = None) -> np.ndarray:
        """
        Linear EOS [kg m⁻³] (Vallis 2006, eq. 1.156): ρ = ρ₀ − α (T − T₀) without S (exactly the phase-2
        expression), ρ = ρ₀ − α (T − T₀) + β (S − S₀) with S [g kg⁻¹] (same shape as T).
        """
        if S is None:
            return self.rho0 - self.t_coef * (T - self.t_ref)
        return self.rho0 - self.t_coef * (T - self.t_ref) + self.s_coef * (S - self.s_ref)

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}
