"""
Equation of state of the 3-D core (N1, role EOS).

Linear in temperature only, as the 3-D core has been since phase 2 (salinity and TEOS-10 enter in N2):

    ρ = ρ₀ − α (T − T₀)   [kg m⁻³]   (Vallis 2006, eq. 1.156, with β S omitted)

ρ₀ [kg m⁻³] reference density, α = ``t_coef`` [kg m⁻³ °C⁻¹], T₀ = ``t_ref`` [°C].
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

    name = "linear_eos"
    role = Role.EOS
    reads = ("T",)
    writes = ()

    def density(self, T: np.ndarray) -> np.ndarray:
        """Linear EOS ρ = ρ₀ − α (T − T₀) [kg m⁻³]."""
        return self.rho0 - self.t_coef * (T - self.t_ref)

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}
