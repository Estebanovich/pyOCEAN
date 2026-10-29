"""
Turbulence closure of the 3-D core (N3 task 1, role CLOSURE; spec 2026-10-29 §2 "El cierre como componente propio").

Until N3 the GLS k-ε lived inside the σ free-surface step (``SigmaWetDry``); here it is a component of its own with
one interface for every closure, so that the GLS in ALE (N3 task 5) and KPP (task 7) plug into the same place.
The physics is unchanged (bit for bit, ``tests/test_closure_component.py``).

Order within a step (CROCO step.F:954-1094, ``step3d_uv2`` then ``gls_mixing``; kept from N2a):

1. ``viscosity(st)``: ν at the u- and v-face interfaces from the **current** closure state; the implicit vertical
   viscosity of the free-surface step uses it.
2. ``update(u, v, h, rho, ustar2_sfc, ustar2_bot)``: after the implicit viscosity and the barotropic correction,
   with the new velocities at the cell centres, the layer thicknesses, the layer density (N² of the stratified GLS)
   and the surface and bed kinematic stresses u*² = |τ|/ρ₀ (σ: the first-layer log law, computed by
   ``SigmaWetDry.closure_forcing`` exactly where it was computed before N3).
3. ``tracer_diffusivity()``: κ at the interfaces for the implicit vertical diffusion of T and S (phase 7), or None:
   the constant ``kv`` as before.

Index order (k, j, i), k = 0 at the bed; interface arrays have N+1 levels, 0 = bed, N = surface.  Inside the GLS
the columns are flattened, (N+1, ny·nx), as in ``physics.mixing.gls.GLSKEpsilon``.

The closure state travels in ``SigmaState.closure`` (references to the closure arrays, no copies): ``load`` puts the
state of the model state into the component before the step, ``state`` gives it back after.

References
----------
Umlauf, L. & Burchard, H. (2003). A generic length-scale equation for geophysical turbulence models. J. Mar. Res. 61,
    235–265.  Port of CROCO gls_mixing.F (``physics/mixing/gls.py``).
"""
from __future__ import annotations

import numpy as np

from core.model.contract import BuildContext, Issue, Role
from physics.mixing.gls import KT_MIN, GLSKEpsilon, buoyancy_frequency2

STATE_PREFIX = "closure_"


class Closure:
    """
    Base turbulence closure: no state, ν and κ left to the constant values of the solvers.

    ``stateful`` False means ``update`` does nothing, so the driver need not build its inputs.
    """
    name = "closure"
    role = Role.CLOSURE
    reads = ("u", "v", "closure")
    writes = ("closure",)
    stateful = False

    def viscosity(self, st) -> tuple[np.ndarray, np.ndarray] | None:
        """ν at the interfaces of the u- and v-faces, (N+1, ny, nx+1) and (N+1, ny+1, nx) [m² s⁻¹], from the
        current closure state; None: the viscosity solver uses its own ν (constant, parabolic or none)."""
        return None

    def update(self, u: np.ndarray, v: np.ndarray, h: np.ndarray, rho: np.ndarray | None,
               ustar2_sfc: np.ndarray | None, ustar2_bot: np.ndarray, bflux: dict | None = None) -> None:
        """
        Advance the closure state with the new velocities.

        u, v : (N, ny, nx) layer velocities at the cell centres after the implicit viscosity [m s⁻¹]
        h : (N, ny, nx) layer thicknesses [m]
        rho : (N, ny, nx) layer density [kg m⁻³] or None
        ustar2_sfc, ustar2_bot : (ny, nx) kinematic surface and bed stress |τ|/ρ₀ [m² s⁻²] (surface None: zero)
        bflux : surface buoyancy fluxes (N3 tasks 6–7), optional
        """
        return None

    def tracer_diffusivity(self) -> np.ndarray | None:
        """κ (N+1, ny, nx) at the interfaces for T and S [m² s⁻¹]; None: the constant ``kv``."""
        return None

    def nonlocal_flux(self) -> np.ndarray | None:
        """Non-local transport at the interfaces (N+1, ny, nx), normalised by the surface flux (KPP), or None."""
        return None

    def remap(self, h_src: np.ndarray, h_dst: np.ndarray) -> None:
        """ALE: carry the closure state from the layers ``h_src`` to ``h_dst`` after the regrid."""
        return None

    def load(self, closure: dict | None) -> None:
        """Put the closure state carried by ``SigmaState.closure`` into the component (by reference)."""
        return None

    def state(self) -> dict | None:
        """The closure state to carry in ``SigmaState.closure`` (references), or None."""
        return None

    def state_fields(self) -> dict[str, np.ndarray]:
        """Restart fields, names prefixed ``closure_``."""
        return {}

    def load_state(self, d: dict[str, np.ndarray]) -> None:
        """Restore the fields of ``state_fields`` (copies)."""
        return None

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []


class ConstantClosure(Closure):
    """
    Closures without prognostic state: 'constant' (ν = nu_const), 'parabolic' (ν = κ u* z (1 − z/D), computed by
    the column solver from its bed u*) and 'none'.  ν stays inside ``SigmaWetDry.column_step`` as before N3.
    """

    def __init__(self, kind: str = "constant") -> None:
        self.kind = kind


class GLSClosure(Closure):
    """
    GLS k-ε closure (Canuto A; Umlauf & Burchard 2003; port of CROCO gls_mixing.F), state at the cell centres.

    Parameters
    ----------
    n_layers : N
    shape : (ny, nx) of the tracer grid
    dt : time step [s]
    z0 : bed roughness length of the bed ψ condition [m] (None: 1e-4, as before)
    stratified : buoyancy (N² from the layer density) and the tracer diffusivity κ_t (N2a task 8c)
    rho0, g : ρ₀ [kg m⁻³] and g [m s⁻²] of N² = −(g/ρ₀) ∂ρ/∂z
    kt_min : floor of κ_t [m² s⁻¹] (CROCO nuws = 1e-6)
    """
    stateful = True

    def __init__(self, n_layers: int, shape: tuple[int, int], dt: float, z0: float | None = None,
                 stratified: bool = False, rho0: float = 1000.0, g: float = 9.81, kt_min: float = KT_MIN) -> None:
        self.n, self.shape, self.dt = int(n_layers), tuple(shape), dt
        self.stratified, self.rho0, self.g = bool(stratified), rho0, g
        ny, nx = self.shape
        if self.stratified:
            self.gls = GLSKEpsilon(self.n, ny * nx, z0_bed=z0 or 1e-4, stratified=True, kt_min=kt_min)
        else:
            self.gls = GLSKEpsilon(self.n, ny * nx, z0_bed=z0 or 1e-4)

    # ------------------------------------------------------------------ interface
    def viscosity(self, st) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre ν (N+1, ny, nx) averaged to u- and v-faces (edge faces: the inner cell; periodic wrap faces,
        ``per``: the mean of cells n−1 and 0)."""
        if st is not None:
            self.load(getattr(st, "closure", None))
        ny, nx = self.shape
        nu = self.gls.akv.reshape(self.n + 1, ny, nx)
        nu_u = np.empty((self.n + 1, ny, nx + 1))
        nu_u[:, :, 1:-1] = 0.5 * (nu[:, :, :-1] + nu[:, :, 1:])
        nu_u[:, :, 0], nu_u[:, :, -1] = nu[:, :, 0], nu[:, :, -1]
        nu_v = np.empty((self.n + 1, ny + 1, nx))
        nu_v[:, 1:-1] = 0.5 * (nu[:, :-1] + nu[:, 1:])
        nu_v[:, 0], nu_v[:, -1] = nu[:, 0], nu[:, -1]
        per = getattr(self, "per", None)
        if per is not None and per.x:
            nu_u[:, :, 0] = nu_u[:, :, -1] = 0.5 * (nu[:, :, -1] + nu[:, :, 0])
        if per is not None and per.y:
            nu_v[:, 0] = nu_v[:, -1] = 0.5 * (nu[:, -1] + nu[:, 0])
        return nu_u, nu_v

    def update(self, u: np.ndarray, v: np.ndarray, h: np.ndarray, rho: np.ndarray | None,
               ustar2_sfc: np.ndarray | None, ustar2_bot: np.ndarray, bflux: dict | None = None) -> None:
        """
        Advance k-ε (``GLSKEpsilon.step``).  Stratified: N² from ``rho`` (``buoyancy_frequency2``, CROCO
        rho_eos.F:404-411) and the surface k = u*²_s/c_μ⁰² with the Charnock roughness; homogeneous: u*²_s = 0
        (``ustar2_sfc`` is not passed, as before N3).
        """
        n = self.n
        dz = np.asarray(h).reshape(n, -1)
        uc, vc = np.asarray(u).reshape(n, -1), np.asarray(v).reshape(n, -1)
        us2b = np.asarray(ustar2_bot).ravel()
        if self.stratified:
            if rho is None:
                raise ValueError("closure='gls' with gls_stratified=True needs the layer density rho in step()")
            n2 = buoyancy_frequency2(np.asarray(rho, float).reshape(n, -1), dz, self.rho0, self.g)
            us2s = None if ustar2_sfc is None else np.asarray(ustar2_sfc).ravel()
            self.gls.step(uc, vc, dz, us2b, self.dt, ustar2_sfc=us2s, n2=n2)
        else:
            self.gls.step(uc, vc, dz, us2b, self.dt)

    def tracer_diffusivity(self) -> np.ndarray | None:
        """Stratified: κ_t = max(c_μ′ k²/ε, kt_min) (N+1, ny, nx), a view of ``gls.akt``; homogeneous: None."""
        if not self.stratified:
            return None
        return self.gls.akt.reshape(self.n + 1, *self.shape)

    def remap(self, h_src: np.ndarray, h_dst: np.ndarray) -> None:
        raise NotImplementedError("the GLS closure in ALE comes with N3 task 5 (remap of h·k and h·ε)")

    def load(self, closure: dict | None) -> None:
        """Load the closure state carried by the model state into the GLS object (by reference, no copies)."""
        gls = self.gls
        if closure is not None:
            gls.tke, gls.eps, gls.akv = closure["tke"], closure["eps"], closure["akv"]
            if "akt" in closure:                                       # stratified GLS (N2a task 8c)
                gls.akt = closure["akt"]

    def state(self) -> dict:
        """The closure state to carry to the next step (references to the GLS arrays)."""
        gls = self.gls
        if gls.stratified:                                             # κ_t is state: phase 7 of the next step
            return {"tke": gls.tke, "eps": gls.eps, "akv": gls.akv, "akt": gls.akt}
        return {"tke": gls.tke, "eps": gls.eps, "akv": gls.akv}

    def state_fields(self) -> dict[str, np.ndarray]:
        return {STATE_PREFIX + k: a for k, a in self.state().items()}

    def load_state(self, d: dict[str, np.ndarray]) -> None:
        gls = self.gls
        shape = gls.tke.shape
        gls.tke, gls.eps, gls.akv = (np.asarray(d[STATE_PREFIX + k]).reshape(shape).copy()
                                     for k in ("tke", "eps", "akv"))
        if gls.stratified:                                             # κ_t of the last step (N2a task 8c)
            gls.akt = np.asarray(d[STATE_PREFIX + "akt"]).reshape(shape).copy()


def make_closure(model) -> Closure:
    """
    The closure component of a ``SigmaBaroclinic`` model, from its core settings: 'gls' → ``GLSClosure``
    (stratified with ``gls_stratified``; κ_t floor ``kv`` if > 0, else CROCO's 1e-6), anything else →
    ``ConstantClosure``.
    """
    core = model.core
    if core.closure != "gls":
        return ConstantClosure(core.closure)
    kv = float(getattr(model, "kv", 0.0))
    return GLSClosure(core.n_layers, core.H.shape, core.dt, z0=core.z0, stratified=core.gls_stratified,
                      rho0=core.rho0, g=core.g, kt_min=kv if kv > 0 else KT_MIN)
