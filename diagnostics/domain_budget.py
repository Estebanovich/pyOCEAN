"""
Volume, heat and salt budget of the whole wet domain of an ALE ``SigmaBaroclinic`` run (regional chain Task 9,
spec 2026-11-01-regional-chain criterion 6).

Each step n → n+1 the change of the domain content must equal the sum of the fluxes that cross its boundary:

    ΔV   = Δt (Q_open + Σ A fw)                                   [m³]
    Δ(VT) = Δt F_open^T + Δt Σ A (Q_ns + Q_sw) / (ρ₀ c_p) + Σ (V_fw − V_adv) T     [°C m³]
    Δ(VS) = Δt F_open^S                                          [g kg⁻¹ m³]

with V = Σ h A over the wet cells (``SigmaBaroclinic.layer_volumes``), Q_open the net layer volume flux into the
domain through the open faces (the layer transports of the step, ``_last_layer_fluxes``), fw the surface
freshwater flux used by the free surface (P − E of the bulk plus river runoff, ``BarotropicFluxes.fw``; real
freshwater: rivers and rain enter with S = 0 and the top-layer T), F_open the step's effective tracer flux
through the open faces (the FCT flux F_L + C·A of ``TracerAdvection``, ``record_open_flux``; Zalesak 1979 eq. 14),
Q_ns, Q_sw the surface heat flux of the step on the wet columns (``Driver.last_bflux``) and Σ (V_fw − V_adv) T the
heat carried by the freshwater volume at the top-layer temperature (``Driver.last_fw_T``).  Every other operator
of the step (remap, vertical diffusion, the KPP non-local term, the shortwave remainder) conserves the content.

The residual of each budget, accumulated over the run, is reported relative to the budget's own terms
(max(|ΔC|, max |accumulated term|), the strictest), to the gross flux Σ_n Σ |terms| and to the content.  The residual
is round-off of the model arithmetic: it scales with the size of the field (salt ≈ 20 × heat on the NWA12 cut), not
with the fluxes, and keeping the budgets in anomaly form Σ V (S − S₀) does not reduce it (checked on the NWA12 cut).

Requires ``vertical_coordinate='ale'`` with the FCT tracer limiter, no tracer sponge band and no interior sponge
(those relax toward the exterior and are not boundary fluxes) and no horizontal tracer diffusion through open faces
(``kh`` = 0).  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import numpy as np

TERMS = {"V": ("open", "surf"), "T": ("open", "surf", "fw"), "S": ("open",)}


class DomainBudget:
    """
    Accumulates the domain budgets of ``model`` step by step (module docstring).

    Usage::

        bud = DomainBudget(m)          # switches the flux records on
        bud.start(st)
        for n in ...:
            st = m.step(st); bud.add(st)
        bud.residuals()                # {'V': {...}, 'T': {...}, 'S': {...}}
    """

    def __init__(self, model) -> None:
        errs = budget_errors(model)
        if errs:
            raise ValueError("DomainBudget: " + "; ".join(errs))
        self.m = model
        model._tracer_advection.record_open_flux = True
        model._driver.record_content = True
        self.sums = {f"{k}_{t}": 0.0 for k, ts in TERMS.items() for t in ts}
        self.gross = {k: 0.0 for k in TERMS}
        self.start_content: dict[str, float] | None = None
        self.content: dict[str, float] | None = None
        self.max_step = {k: 0.0 for k in TERMS}            # largest |residual| of one step, relative to its gross
        self.n_steps = 0

    # ------------------------------------------------------------------ contents and fluxes
    def contents(self, st) -> dict[str, float]:
        """Σ V, Σ V T, Σ V S over the domain [m³, °C m³, g kg⁻¹ m³]."""
        V = np.nan_to_num(self.m.layer_volumes(st.eta, st.h))
        out = {"V": float(V.sum()), "T": float(np.sum(np.where(V > 0, V * st.T, 0.0)))}
        out["S"] = float(np.sum(np.where(V > 0, V * st.S, 0.0))) if st.S is not None else 0.0
        return out

    def step_fluxes(self) -> dict[str, float]:
        """The boundary terms of the last step [content per step], from the model's records."""
        m, dt = self.m, self.m.dt
        lf = m._last_layer_fluxes
        q = 0.0
        for edge, Q, sl, sgn in (("west", lf.Qx, np.s_[..., 0], 1.0), ("east", lf.Qx, np.s_[..., -1], -1.0),
                                 ("south", lf.Qy, np.s_[:, 0, :], 1.0), ("north", lf.Qy, np.s_[:, -1, :], -1.0)):
            if edge in m.open:
                q += sgn * float(np.nansum(Q[sl]))
        fw = getattr(m._fluxes, "fw", None)
        area = np.nan_to_num(m.area)
        surf_v = 0.0 if fw is None else dt * float(np.sum(area * np.nan_to_num(fw)))
        bf = m._driver.last_bflux
        heat = 0.0
        if bf is not None:
            Q = np.nan_to_num(bf["Q_ns"]) + (0.0 if bf.get("Q_sw") is None else np.nan_to_num(bf["Q_sw"]))
            heat = dt * float(np.sum(area * Q)) / (m.rho0 * m.cp)
        F = m._tracer_advection.last_open_flux
        lay = _layout(m)
        out = {"V_open": dt * q, "V_surf": surf_v,
               "T_open": dt * float(F[lay["T"]]), "T_surf": heat, "T_fw": float(m._driver.last_fw_T),
               "S_open": dt * float(F[lay["S"]]) if "S" in lay else 0.0}
        return out

    # ------------------------------------------------------------------ accumulation
    def start(self, st) -> None:
        """Content at the start of the accumulation (the state before the first ``add``)."""
        self.start_content = self.contents(st)
        self.content = dict(self.start_content)

    def add(self, st_new) -> dict[str, float]:
        """Accumulate one step ending in ``st_new``; returns that step's residual per budget [content]."""
        if self.content is None:
            raise ValueError("DomainBudget.start(st) must be called before the first add")
        f = self.step_fluxes()
        new = self.contents(st_new)
        res = {}
        for k, ts in TERMS.items():
            terms = [f[f"{k}_{t}"] for t in ts]
            for t, v in zip(ts, terms):
                self.sums[f"{k}_{t}"] += v
            g = sum(abs(v) for v in terms)
            self.gross[k] += g
            res[k] = (new[k] - self.content[k]) - sum(terms)
            if g > 0:
                self.max_step[k] = max(self.max_step[k], abs(res[k]) / g)
        self.content = new
        self.n_steps += 1
        return res

    def residuals(self) -> dict[str, dict[str, float]]:
        """
        Per budget: the content change, the accumulated terms, the residual and its size relative to the budget's own
        terms, max(|change|, max |accumulated term|) (``rel_terms``, the strictest), to the gross flux Σ_n Σ |terms|
        (``rel_gross``; the oscillating tidal open-face flux inflates it) and to the start content (``rel_content``);
        ``max_step_rel`` is the worst step relative to its gross flux.
        """
        out = {}
        for k, ts in TERMS.items():
            change = self.content[k] - self.start_content[k]
            terms = {t: self.sums[f"{k}_{t}"] for t in ts}
            r = change - sum(terms.values())
            scale = max([abs(change)] + [abs(v) for v in terms.values()])
            out[k] = dict(change=change, **terms, residual=r,
                          rel_terms=abs(r) / scale if scale > 0 else 0.0,
                          rel_gross=abs(r) / self.gross[k] if self.gross[k] > 0 else 0.0,
                          rel_content=abs(r) / abs(self.start_content[k]) if self.start_content[k] else 0.0,
                          max_step_rel=self.max_step[k], gross=self.gross[k])
        return out

    # ------------------------------------------------------------------ restart of the accumulators
    def state(self) -> dict[str, np.ndarray]:
        """Accumulators as float64 arrays (``np.savez``), for a run split by a restart."""
        d = {f"sum_{k}": np.array(v) for k, v in self.sums.items()}
        d.update({f"gross_{k}": np.array(v) for k, v in self.gross.items()})
        d.update({f"max_{k}": np.array(v) for k, v in self.max_step.items()})
        d.update({f"start_{k}": np.array(v) for k, v in self.start_content.items()})
        d.update({f"content_{k}": np.array(v) for k, v in self.content.items()})
        d["n_steps"] = np.array(self.n_steps)
        return d

    def load(self, d) -> None:
        """Restore ``state()``."""
        self.sums = {k: float(d[f"sum_{k}"]) for k in self.sums}
        self.gross = {k: float(d[f"gross_{k}"]) for k in self.gross}
        self.max_step = {k: float(d[f"max_{k}"]) for k in self.max_step}
        self.start_content = {k: float(d[f"start_{k}"]) for k in TERMS}
        self.content = {k: float(d[f"content_{k}"]) for k in TERMS}
        self.n_steps = int(d["n_steps"])


def _layout(m) -> dict[str, int]:
    from core.model.driver import _stack_layout
    return _stack_layout(m)


def budget_errors(m) -> list[str]:
    """Configurations whose boundary terms the budget does not cover (module docstring)."""
    out = []
    if not getattr(m, "ale", False):
        out.append("needs vertical_coordinate='ale'")
    if getattr(m, "tracer_limiter", None) != "fct":
        out.append("needs the FCT tracer limiter (its open-face fluxes are recorded)")
    if getattr(m, "_sp_c", None) is not None and getattr(m, "sponge_tracer", False):
        out.append("a tracer sponge band (sponge_cells > 0) relaxes toward the exterior: not a boundary flux")
    if getattr(m, "sponge_rate", None) is not None:
        out.append("an interior sponge (sponge_rate) is not a boundary flux")
    if getattr(m, "kh", 0.0) > 0:
        out.append("kh > 0: along-layer diffusion through open faces is not recorded")
    if getattr(m, "periodic", None):
        out.append("periodic axes are not supported")
    return out
