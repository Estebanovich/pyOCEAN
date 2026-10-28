"""
Exact discrete budget of the tracer "variance" Σ V c² of S and T, operator by operator (N2a task 6, spec
docs/superpowers/specs/2026-10-28-n2a-salinity-freshwater-design.md §5; ``SigmaBaroclinic(variance_budget=True)``).

Discrete variance decay (DVD; Klingbeil et al. 2014, Ocean Modell. 81, 49-64, doi:10.1016/j.ocemod.2014.06.001,
§2.2; local library ``Klingbeil_2014 - Quantification of Spurious Dissipation and Mixing``): each operator that
changes a tracer c from c_a to c_b on volumes V changes the second moment Σ V c² by a per-cell amount, and the sum
of all of them is the tendency of Σ V c² to round-off.  Per cell, in [c² m³ s⁻¹] (S: (g kg⁻¹)² m³ s⁻¹, T: °C² m³ s⁻¹):

  ``{c}_num_adv``    V¹ (⟨c²⟩_adv − c_adv²)/Δt — numerical mixing of the 3-D advection (SSP-RK3 / FCT / 'pc',
                     the step's effective face fluxes), with c² advected in the same stack (Burchard & Rennau 2008,
                     Ocean Modell. 20, 293-311, eq. 24).  Computed right after the advection, before the freshwater
                     substep (the substep scales the S² member by r while S² scales by r², so it must not enter);
  ``{c}_surf``       freshwater substep (task 5): (V_fw c_new² − V_adv c_old²)/Δt; S: ``var_surf`` of
                     ``core.components.freshwater.apply_freshwater`` (Lorenz et al. 2021 eq. 43); T keeps its
                     value, so (V_fw − V_adv) T²/Δt.  A source (> 0 increases Σ V c²);
  ``{c}_phy_h``      V (c_a² − c_b²)/Δt of the horizontal (along-layer) K_h diffusion;
  ``{c}_phy_v``      V (c_a² − c_b²)/Δt of the implicit vertical K_v diffusion;
  ``{c}_num_remap``  ALE only: V_new (⟨c²⟩_remap − c_new²)/Δt with c² remapped with the stack (as ``mix_num``);
  ``{c}_sponge``     V (c_a² − c_b²)/Δt of the edge tracer sponge (on the advected volumes) and of the ALE interior
                     sponge (on the new volumes).

Mixing terms (num_adv, phy_h, phy_v, num_remap) are positive when they destroy variance; sponges are counted the same
way but apart; open-boundary advective fluxes of c² are not a separate key (they are the residual of the closure).
In a closed basin, per step,

    Σ V¹ c¹² − Σ V⁰ c⁰² = Δt Σ (surf − num_adv − phy_h − phy_v − num_remap − sponge)

because the advection and the remap conserve the content of the c² member, and every other term is the exact change
of Σ V c² across its operator.  Compared with ``mix_num`` (T only): in σ, ``T_num_adv`` = ``mix_num``; in ALE,
``T_num_adv + T_num_remap`` = ``mix_num`` (same formulas, kept as the compatibility diagnostic).

Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import numpy as np

TRACERS = ("S", "T")
TERMS = ("num_adv", "surf", "phy_h", "phy_v", "sponge")


class VarianceBudget:
    """Per-step accumulator of the variance terms of S and T, (N, ny, nx) each (module docstring)."""

    def __init__(self, shape: tuple[int, ...], dt: float, ale: bool) -> None:
        self.dt = float(dt)
        terms = TERMS + (("num_remap",) if ale else ())
        self.terms: dict[str, np.ndarray] = {f"{c}_{t}": np.zeros(shape) for c in TRACERS for t in terms}

    def advective(self, c: str, V1: np.ndarray, c2_adv: np.ndarray, c_adv: np.ndarray) -> None:
        """Numerical mixing of the advection: V¹ (⟨c²⟩ − c²)/Δt (Burchard & Rennau 2008 eq. 24)."""
        self.terms[f"{c}_num_adv"] += V1 * (c2_adv - c_adv**2) / self.dt

    def remap(self, c: str, V_new: np.ndarray, c2_rem: np.ndarray, c_rem: np.ndarray) -> None:
        """Numerical mixing of the ALE remap: V_new (⟨c²⟩ − c²)/Δt (as ``mix_num``)."""
        self.terms[f"{c}_num_remap"] += V_new * (c2_rem - c_rem**2) / self.dt

    def operator(self, key: str, V: np.ndarray, before: np.ndarray, after: np.ndarray) -> None:
        """Variance destroyed by an operator that maps ``before`` to ``after`` on fixed volumes V: V (a² − b²)/Δt."""
        self.terms[key] += V * (before**2 - after**2) / self.dt

    def source(self, key: str, value: np.ndarray) -> None:
        """A source of Σ V c² [c² m³ s⁻¹] (the freshwater term)."""
        self.terms[key] += value
