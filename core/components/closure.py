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
from core.vertical import remap as rm
from physics.mixing.gls import KT_MIN, N2_FLOOR, RM, RN, RP, GLSKEpsilon, buoyancy_frequency2
from physics.mixing.kpp_cvmix import KPPParams, kpp_column

STATE_PREFIX = "closure_"
H_MASSLESS = 1e-3          # [m] ALE: thinner layers are merged for the GLS column (HYCOM onemm, mxkprf.F90:1866-1878)
PRODUCTIONS = ("croco", "energy")


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
    update_phase = "implicit"     # phase whose driver calls ``update``: 'implicit' (4, GLS) or 'tracers' (7, KPP)
    restart_prefix = ""           # file-variable prefix of the restart state (``core.restart``)

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

    def bind_ale(self, remap_code: int) -> None:
        """ALE mode: the layers are the model's h (not D Δσ) and the state is remapped after each regrid."""
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
    GLS k-ε closure (Canuto A; Umlauf & Burchard 2003; port of CROCO gls_mixing.F), state at the layer interfaces.

    Parameters
    ----------
    n_layers : N
    shape : (ny, nx) of the tracer grid
    dt : time step [s]
    z0 : bed roughness length of the bed ψ condition [m] (None: 1e-4, as before)
    stratified : buoyancy (N² from the layer density) and the tracer diffusivity κ_t (N2a task 8c)
    rho0, g : ρ₀ [kg m⁻³] and g [m s⁻²] of N² = −(g/ρ₀) ∂ρ/∂z
    kt_min : floor of κ_t [m² s⁻¹] (CROCO nuws = 1e-6)
    production : shear production P of the k and ψ equations (N3 task 5, spec decision 5):
        'croco' (default) — P = ν_old S² with the new shear (CROCO gls_mixing.F:621), bit for bit as before;
        'energy' — the energy-conserving form of Burchard (2002, Ocean Modell. 4, 347–361, eq. 16), given by the
        driver as the kinetic energy that the implicit vertical viscosity removed at each interface (``shear_work``,
        ``core.model.driver``), divided by the interface cell height ½(h_k−1 + h_k) of the k equation.
    h_massless : ALE only: layers thinner than this [m] are merged for the closure column (``update``).

    ALE mode (``bind_ale``, N3 task 5; spec §3 and decisions 2–4)
    -------------------------------------------------------------
    * **Thin layers** (``_update_ale``): before N², S² and the solve, every layer with h < ``h_massless`` is merged
      with the nearest layer with mass above it (below it at the top of the column), with thickness-weighted u, v
      and ρ — MOM6 ``eliminate_massless`` (MOM_kappa_shear.F90:254-307: a massless layer joins the cluster of the
      layer above it, sums of h, u·h, T·h).  The GLS is then solved on the merged column (the same
      ``GLSKEpsilon`` code as σ) and its interface values are put back on every model interface by linear
      interpolation in the fractional height within the merged layer (MOM_kappa_shear.F90:355-363, ``kf``), so an
      interface between a vanished layer and its neighbour takes the value of the merged interface it sits on.  A
      vanished layer can therefore not create an N² or shear spike (its Δz never enters), and the bed distance of
      the log-layer ψ flux is the first layer with mass.  The threshold 1 mm is HYCOM's ``onemm`` (mxkprf.F90:
      1866-1878 with MASSLESS_1MM; MOM6 uses 0.1 √(Δt κ₀), MOM_kappa_shear.F90:221, ≈ 8 mm for Δt = 600 s).  A
      column with one merged layer has no interior interface: k, ε, ν, κ_t go to their floors.  Without thin layers
      the GLS is stepped on the model columns directly (no merging arithmetic: σ ≡ ALE with fixed layers).
    * **Remap** (``remap``): k, ε, ν, κ_t (and the N² of the last update) live at the interfaces; each interface i
      owns the dual cell between the centres of layers i−1 and i, w_i = ½(h_i−1 + h_i) (half layers at the bed and
      surface), which tile the column.  After the regrid the interface values are remapped as cell means of the dual
      grid, from w(h_src) to w(h_dst), with the model's conservative remap (``core.vertical.remap``, PPM_H4 by
      default, monotone limiter) — the dual-grid ("inverse") reading of interface quantities of Halliwell (2004,
      Fig. 2) and the thickness-weighted carry of HYCOM's q², q²l (hybgen.F90:961-962, 1463-1479) and
      Oceananigans' e·σ (``docs/design/n2c_refs/tke_in_ale.md``).  Σ w k is conserved to round-off and the limiter
      creates no new extrema, so k, ε stay ≥ their floors; the floors (tke_min 1e-10, eps_min 1e-12, ν_min, κ_min)
      and the Galperin limit l ≤ 0.53 √(2k)/N (gls_mixing.F:829-834) with the remapped N² are applied after.  An
      unchanged grid (h_src == h_dst) is not remapped.
    """
    stateful = True
    restart_prefix = "gls_"       # file names of the restarts written before N3 (``core.restart``)

    def __init__(self, n_layers: int, shape: tuple[int, int], dt: float, z0: float | None = None,
                 stratified: bool = False, rho0: float = 1000.0, g: float = 9.81, kt_min: float = KT_MIN,
                 production: str = "croco", h_massless: float = H_MASSLESS) -> None:
        self.n, self.shape, self.dt = int(n_layers), tuple(shape), dt
        self.stratified, self.rho0, self.g = bool(stratified), rho0, g
        if production not in PRODUCTIONS:
            raise ValueError(f"unknown gls_production {production!r}; use one of {PRODUCTIONS}")
        self.production, self.h_massless = production, float(h_massless)
        self.z0 = z0
        self.ale, self.remap_code = False, None
        self._n2 = None                                                # N² (N+1, M) of the last update, for remap
        ny, nx = self.shape
        if self.stratified:
            self.gls = GLSKEpsilon(self.n, ny * nx, z0_bed=z0 or 1e-4, stratified=True, kt_min=kt_min)
        else:
            self.gls = GLSKEpsilon(self.n, ny * nx, z0_bed=z0 or 1e-4)

    def bind_ale(self, remap_code: int) -> None:
        self.ale, self.remap_code = True, int(remap_code)

    # ------------------------------------------------------------------ interface
    def viscosity(self, st) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre ν (N+1, ny, nx) averaged to u- and v-faces (edge faces: the inner cell; periodic wrap faces,
        ``per``: the mean of cells n−1 and 0)."""
        if st is not None:
            self.load(getattr(st, "closure", None))
        return centre_to_faces(self.gls.akv.reshape(self.n + 1, *self.shape), getattr(self, "per", None))

    def update(self, u: np.ndarray, v: np.ndarray, h: np.ndarray, rho: np.ndarray | None,
               ustar2_sfc: np.ndarray | None, ustar2_bot: np.ndarray, bflux: dict | None = None,
               shear_work: np.ndarray | None = None) -> None:
        """
        Advance k-ε (``GLSKEpsilon.step``).  Stratified: N² from ``rho`` (``buoyancy_frequency2``, CROCO
        rho_eos.F:404-411) and the surface k = u*²_s/c_μ⁰² with the Charnock roughness; homogeneous: u*²_s = 0
        (``ustar2_sfc`` is not passed, as before N3).

        shear_work : (N−1, ny, nx) kinetic energy per unit area removed by the implicit vertical viscosity at the
            interior interfaces during the step [m³ s⁻³] (production 'energy', Burchard 2002 eq. 16 times the
            centre distance; ``core.model.driver``), None with 'croco'.
        ALE (``bind_ale``): thin layers merged first (class docstring).
        """
        n = self.n
        dz = np.asarray(h).reshape(n, -1)
        uc, vc = np.asarray(u).reshape(n, -1), np.asarray(v).reshape(n, -1)
        us2b = np.asarray(ustar2_bot).ravel()
        if self.stratified and rho is None:
            raise ValueError("closure='gls' with gls_stratified=True needs the layer density rho in step()")
        rh = None if rho is None else np.asarray(rho, float).reshape(n, -1)
        us2s = None if (ustar2_sfc is None or not self.stratified) else np.asarray(ustar2_sfc).ravel()
        work = None if shear_work is None else np.asarray(shear_work, float).reshape(n - 1, -1)
        if self.ale and not (dz >= self.h_massless).all():
            self._update_ale(uc, vc, dz, rh, us2s, us2b, work)
            return
        prod = None if work is None else work / (0.5 * (dz[:-1] + dz[1:]))
        if self.stratified:
            n2 = buoyancy_frequency2(rh, dz, self.rho0, self.g)
            self.gls.step(uc, vc, dz, us2b, self.dt, ustar2_sfc=us2s, n2=n2, prod=prod)
            if self.ale:
                self._n2 = _pad_interfaces(n2)
        else:
            self.gls.step(uc, vc, dz, us2b, self.dt, prod=prod)

    def _update_ale(self, uc: np.ndarray, vc: np.ndarray, dz: np.ndarray, rho: np.ndarray | None,
                    us2s: np.ndarray | None, us2b: np.ndarray, work: np.ndarray | None) -> None:
        """
        GLS step on columns with thin layers (class docstring, MOM6 eliminate_massless): merge, solve the columns
        with the same number c of merged layers together, interpolate back to the model interfaces.
        """
        gls, n = self.gls, self.n
        act = dz >= self.h_massless                                    # layers with mass, (N, M)
        cnt = act.sum(axis=0)
        k_idx = np.arange(n)[:, None]
        # owner layer of each layer: the nearest layer with mass above it (k' ≥ k), else the nearest below
        above = np.minimum.accumulate(np.where(act, k_idx, n)[::-1], axis=0)[::-1]
        below = np.maximum.accumulate(np.where(act, k_idx, -1), axis=0)
        owner = np.where(above < n, above, below)
        rank = np.cumsum(act, axis=0) - 1                              # merged index of each layer with mass
        tke, eps, akv = gls.tke.copy(), gls.eps.copy(), gls.akv.copy()
        akt = gls.akt.copy() if self.stratified else None
        n2_full = np.full((n + 1, dz.shape[1]), N2_FLOOR)
        last_prod = np.zeros((n - 1, dz.shape[1]))
        for c in np.unique(cnt):
            cols = np.flatnonzero(cnt == c)
            if c == 0:                                                 # dry or massless column: unchanged
                continue
            if c == 1:                                                 # no interior interface: floors
                tke[:, cols], eps[:, cols], akv[:, cols] = gls.tke_min, gls.psi_min, gls.nu_min
                if akt is not None:
                    akt[:, cols] = gls.kt_min
                continue
            mg = cols.size
            lay = np.take_along_axis(rank[:, cols], np.clip(owner[:, cols], 0, n - 1), axis=0)   # merged layer of each layer
            colix = np.broadcast_to(np.arange(mg), (n, mg))
            h_s = dz[:, cols]

            def merge(x: np.ndarray) -> np.ndarray:
                out = np.zeros((c, mg))
                np.add.at(out, (lay, colix), x)
                return out
            hc = merge(h_s)
            uc_c, vc_c = merge(h_s * uc[:, cols]) / hc, merge(h_s * vc[:, cols]) / hc
            # fractional height of each model interface inside its merged layer (MOM6 kc, kf)
            zf = np.concatenate([np.zeros((1, mg)), np.cumsum(h_s, axis=0)])
            zc = np.concatenate([np.zeros((1, mg)), np.cumsum(hc, axis=0)])
            kc = np.concatenate([lay, lay[-1:]])                       # interface i: layer i (top: layer N−1)
            kf = np.clip((zf - np.take_along_axis(zc, kc, axis=0)) / np.take_along_axis(hc, kc, axis=0), 0.0, 1.0)
            kf[0] = 0.0
            kf[-1] = 1.0
            # the merged interface j is the bottom of merged layer j (j < c) and the surface (j = c)
            first = np.zeros((c + 1, mg), int)
            nl = merge(np.ones((n, mg)))
            first[1:c] = np.cumsum(nl, axis=0)[:-1].astype(int)
            first[c] = n

            def gather(x: np.ndarray) -> np.ndarray:
                return np.take_along_axis(x[:, cols], first, axis=0)

            def scatter(xc: np.ndarray) -> np.ndarray:
                lo = np.take_along_axis(xc, kc, axis=0)
                hi = np.take_along_axis(xc, np.minimum(kc + 1, c), axis=0)
                mix = np.clip((1.0 - kf) * lo + kf * hi, np.minimum(lo, hi), np.maximum(lo, hi))   # stays ≥ floors
                return np.where(kf == 0.0, lo, np.where(kf == 1.0, hi, mix))

            sub = GLSKEpsilon(int(c), mg, z0_bed=gls.z0_bed, stratified=self.stratified, kt_min=gls.kt_min)
            sub.tke, sub.eps, sub.akv = gather(gls.tke), gather(gls.eps), gather(gls.akv)
            if self.stratified:
                sub.akt = gather(gls.akt)
            prod = None
            if work is not None:                                       # energy split as the interpolation weights
                wf = np.zeros((n + 1, mg))
                wf[1:-1] = work[:, cols]
                ec = np.zeros((c + 1, mg))
                np.add.at(ec, (kc, colix[:1].repeat(n + 1, 0)), (1.0 - kf) * wf)
                np.add.at(ec, (np.minimum(kc + 1, c), colix[:1].repeat(n + 1, 0)), kf * wf)
                prod = ec[1:-1] / (0.5 * (hc[:-1] + hc[1:]))
            us2s_c = None if us2s is None else us2s[cols]
            if self.stratified:
                n2 = buoyancy_frequency2(merge(h_s * rho[:, cols]) / hc, hc, self.rho0, self.g)
                sub.step(uc_c, vc_c, hc, us2b[cols], self.dt, ustar2_sfc=us2s_c, n2=n2, prod=prod)
                n2_full[:, cols] = scatter(_pad_interfaces(n2))
                akt[:, cols] = scatter(sub.akt)
            else:
                sub.step(uc_c, vc_c, hc, us2b[cols], self.dt, prod=prod)
            tke[:, cols], eps[:, cols], akv[:, cols] = scatter(sub.tke), scatter(sub.eps), scatter(sub.akv)
            last_prod[:, cols] = scatter(_pad_interfaces(sub.last_prod))[1:-1]
        gls.tke, gls.eps, gls.akv = tke, eps, akv
        if akt is not None:
            gls.akt = akt
        gls.last_prod = last_prod
        self._n2 = n2_full if self.stratified else None

    def tracer_diffusivity(self) -> np.ndarray | None:
        """Stratified: κ_t = max(c_μ′ k²/ε, kt_min) (N+1, ny, nx), a view of ``gls.akt``; homogeneous: None."""
        if not self.stratified:
            return None
        return self.gls.akt.reshape(self.n + 1, *self.shape)

    def remap(self, h_src: np.ndarray, h_dst: np.ndarray) -> None:
        """
        Carry k, ε, ν, κ_t from the layers ``h_src`` to ``h_dst`` (N, ny, nx) [m] after the ALE regrid: dual-grid
        conservative remap, floors and the Galperin limit (class docstring).
        """
        if self.remap_code is None:
            raise RuntimeError("GLSClosure.remap needs the ALE mode (bind_ale)")
        gls, n = self.gls, self.n
        hs = np.asarray(h_src, float).reshape(n, *self.shape)
        hd = np.asarray(h_dst, float).reshape(n, *self.shape)
        if np.array_equal(hs, hd):
            return
        ws, wd = dual_thickness(hs), dual_thickness(hd)
        wet = (ws.sum(axis=0) > 0) & (wd.sum(axis=0) > 0)
        shp = (n + 1,) + self.shape

        def rmp(x: np.ndarray) -> np.ndarray:
            return rm.remap_columns(ws, x.reshape(shp), wd, self.remap_code, wet).reshape(n + 1, -1)
        tke = np.maximum(rmp(gls.tke), gls.tke_min)
        eps = np.maximum(rmp(gls.eps), gls.psi_min)
        gls.akv = np.maximum(rmp(gls.akv), gls.nu_min)
        if self.stratified:
            gls.akt = np.maximum(rmp(gls.akt), gls.kt_min)
        n2 = N2_FLOOR if self._n2 is None else rmp(self._n2)[1:-1]
        k = tke[1:-1]                                                  # Galperin l ≤ 0.53 √(2k)/N (gls_mixing.F:829-834)
        l_lim = gls.galp * np.sqrt(2.0 * k) / np.sqrt(np.maximum(N2_FLOOR, n2))
        eps[1:-1] = np.maximum(eps[1:-1], gls.cm0**RP * l_lim**RN * k**RM)
        gls.tke, gls.eps = tke, eps
        self._n2 = None

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


def centre_to_faces(nu: np.ndarray, per=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Interface values at the cell centres (N+1, ny, nx) averaged to the u- and v-faces, (N+1, ny, nx+1) and
    (N+1, ny+1, nx): the mean of the two cells; edge faces take the inner cell; periodic wrap faces (``per``) the
    mean of cells n−1 and 0 (CROCO gls_mixing.F averages Akv to the faces the same way).
    """
    n1, ny, nx = nu.shape
    nu_u = np.empty((n1, ny, nx + 1))
    nu_u[:, :, 1:-1] = 0.5 * (nu[:, :, :-1] + nu[:, :, 1:])
    nu_u[:, :, 0], nu_u[:, :, -1] = nu[:, :, 0], nu[:, :, -1]
    nu_v = np.empty((n1, ny + 1, nx))
    nu_v[:, 1:-1] = 0.5 * (nu[:, :-1] + nu[:, 1:])
    nu_v[:, 0], nu_v[:, -1] = nu[:, 0], nu[:, -1]
    if per is not None and per.x:
        nu_u[:, :, 0] = nu_u[:, :, -1] = 0.5 * (nu[:, :, -1] + nu[:, :, 0])
    if per is not None and per.y:
        nu_v[:, 0] = nu_v[:, -1] = 0.5 * (nu[:, -1] + nu[:, 0])
    return nu_u, nu_v


def dual_thickness(h: np.ndarray) -> np.ndarray:
    """
    Thickness of the dual cells of the interfaces (N+1, ...) from the layers h (N, ...) [m]: interface i owns
    ½ h_{i−1} + ½ h_i (h_{−1} = h_N = 0), so Σ_i w_i = Σ_k h_k and Σ_i w_i k_i is the column integral of an
    interface quantity (the TKE cell heights of ``GLSKEpsilon._solve`` in the interior).
    """
    w = np.zeros((h.shape[0] + 1,) + h.shape[1:])
    w[:-1] += 0.5 * h
    w[1:] += 0.5 * h
    return w


def _pad_interfaces(x: np.ndarray) -> np.ndarray:
    """(N−1, M) interior-interface values → (N+1, M) with the bed and surface values copied from their neighbours."""
    return np.concatenate([x[:1], x, x[-1:]]) if x.shape[0] else np.full((2,) + x.shape[1:], N2_FLOOR)


class KPPClosure(Closure):
    """
    K-profile parameterisation (Large, McWilliams & Doney 1994) as CVMix computes it for MOM6
    (``physics.mixing.kpp_cvmix.kpp_column``; N3 task 7, spec §4 and decisions 6–7), in σ and ALE.

    Parameters
    ----------
    n_layers : N
    shape : (ny, nx) of the tracer grid
    params : ``KPPParams`` (None: the CVMix/MOM6 defaults, no background ν or κ)
    rho0, g : ρ₀ [kg m⁻³], g [m s⁻²] of the buoyancy b = −g (ρ − ρ₀)/ρ₀
    t_coef, s_coef : the linear EOS ρ = ρ₀ − α (T − T₀) + β (S − S₀): α [kg m⁻³ °C⁻¹], β [kg m⁻³ (g kg⁻¹)⁻¹]
    cp : heat capacity c_p [J kg⁻¹ K⁻¹] of the heat-flux conversion (the model's ``cp``)

    Place in the step (MOM6 diabatic_ALE_legacy, MOM_diabatic_driver.F90:784-848, 991, 1053-1059; read in
    ``docs/design/n2c_refs/kpp_mom6_cvmix.md`` §1): ``update`` is called by the driver in **phase 7** (tracers),
    after the advection and the freshwater substep and before the surface heat flux enters T, with the layer
    thicknesses, velocities and density of that moment (MOM6 computes K and γ from the state at the start of the
    diabatic step; the only difference is that pyOCEAN's along-layer K_h comes after, MOM6's before).  Then

    1. the non-local term redistributes the surface fluxes (``nonlocal_flux`` × F_T, F_S; the driver applies it
       explicitly, MOM_CVMix_KPP.F90:1899-1907 — it changes no column content);
    2. the surface heat flux heats T (N3 task 2);
    3. κ_s (``tracer_diffusivity``) diffuses T, S and passive tracers implicitly, as the GLS κ_t;
    4. ν (``viscosity``) is the implicit vertical viscosity of the **next** step's momentum (MOM6: Kv_KPP is added
       to visc%Kv_shear and used by the next dynamics step, MOM_CVMix_KPP.F90:1015-1021).  ν is therefore the only
       state carried between steps (``SigmaState.closure['akv']``, restart variable ``kpp_akv``); h, κ_s and γ are
       recomputed every step, as in MOM6 (OBLdepth is in MOM6's restart only for FPMIX, MOM_CVMix_KPP.F90:241-243).

    Surface forcing (spec decision 6: for KPP the heat flux enters as a surface buoyancy flux), positive =
    stabilising (MOM6/CVMix, cvmix_kpp.F90:821):

        F_T = Q_ns / (ρ₀ c_p) [K m s⁻¹],      F_S = −S₁ (P − E) [g kg⁻¹ m s⁻¹] (salinity and ``fw`` only),
        B_ns = (g/ρ₀)(α F_T − β F_S),          B_sw = (g/ρ₀) α Q_sw / (ρ₀ c_p)

    with S₁ the top-layer salinity: pyOCEAN's freshwater is a real volume flux, and its effect on the surface
    buoyancy is the dilution −S₁ (P − E), the term MOM6 adds to netSalt for KPP (MOM_forcing_type.F90:1028-1151).
    Only the shortwave absorbed above the trial depth enters B_f (L94 A3–A4 through CVMix's
    ``buoyFlux(1) − buoyFlux(k+1)``, MOM_CVMix_KPP.F90:1383, and ``KPP_SHORTWAVE_METHOD = MXL_SW`` for K and γ,
    837-845), with the Paulson & Simpson fractions ``sw_frac_iface`` of the step.  u* = (|τ_s|/ρ₀)^{1/2} from the
    wind stress (MOM6 ``find_ustar``, MOM_forcing_type.F90:1187-1249).  The non-local term uses the non-solar fluxes
    F_T, F_S (MOM6 KPP_temp_flux = net heat minus SW, KPP_salt_flux; MOM_diabatic_driver.F90:845-848).

    Columns: only wet columns (``h`` summing to > 0; the driver zeroes the dry and land columns, as the heat flux)
    are computed; elsewhere ν = κ = γ = 0.  The bed (k = 0) and surface (k = N) interface values are set to zero:
    with h reaching the bed CVMix writes the enhanced K and γ at the bed interface (Task 6), which must not become
    a flux through the bed (the implicit solver ignores them; γ would break the conservation of the column).

    ALE: ν is kept by interface index through the regrid, as MOM6 keeps visc%Kv_shear (only halo-updated in
    ALE_regridding_and_remapping, MOM.F90:1964-1965); the KPP state is not remapped.
    Interior Richardson-number mixing (L94 §4) is not part of 'kpp' (MOM6 adds it from its own interior schemes;
    spec item 4 leaves it optional, decided here: off).  Index order (k, j, i), k = 0 at the bed; interfaces N+1.
    """
    stateful = True
    update_phase = "tracers"
    restart_prefix = "kpp_"

    def __init__(self, n_layers: int, shape: tuple[int, int], params: KPPParams | None = None,
                 rho0: float = 1000.0, g: float = 9.81, t_coef: float = 0.2, s_coef: float = 0.0,
                 cp: float = 3985.0) -> None:
        self.n, self.shape = int(n_layers), tuple(shape)
        self.params = KPPParams() if params is None else params
        self.rho0, self.g, self.t_coef, self.s_coef, self.cp = float(rho0), float(g), float(t_coef), float(s_coef), \
            float(cp)
        m = self.shape[0] * self.shape[1]
        self.akv = np.zeros((self.n + 1, m))                          # ν of the last update [m² s⁻¹]
        self.akt = np.zeros((self.n + 1, m))                          # κ_s of the last update [m² s⁻¹]
        self.gamma = np.zeros((self.n + 1, m))                        # γ of the last update [1]
        self.obl_depth = np.zeros(m)                                  # h [m]
        self.F_T = np.zeros(m)                                        # non-solar F_T [K m s⁻¹]
        self.F_S = np.zeros(m)                                        # F_S [g kg⁻¹ m s⁻¹]
        self.f = None                                                 # Coriolis at the centres (Ekman limit)

    def viscosity(self, st) -> tuple[np.ndarray, np.ndarray]:
        """ν of the last update averaged to the u- and v-faces (``centre_to_faces``)."""
        if st is not None:
            self.load(getattr(st, "closure", None))
        return centre_to_faces(self.akv.reshape(self.n + 1, *self.shape), getattr(self, "per", None))

    def buoyancy_fluxes(self, Q_ns: np.ndarray, Q_sw: np.ndarray | None, fw: np.ndarray | None,
                        S_sfc: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Kinematic surface fluxes and buoyancy fluxes (class docstring), all (M,):
        F_T [K m s⁻¹], F_S [g kg⁻¹ m s⁻¹], B_ns, B_sw [m² s⁻³] (positive = stabilising).
        """
        rc = self.rho0 * self.cp
        F_T = np.asarray(Q_ns, float).ravel() / rc
        F_S = np.zeros_like(F_T) if (fw is None or S_sfc is None) else \
            -np.asarray(S_sfc, float).ravel() * np.asarray(fw, float).ravel()
        go = self.g / self.rho0
        B_ns = go * (self.t_coef * F_T - self.s_coef * F_S)
        B_sw = np.zeros_like(F_T) if Q_sw is None else go * self.t_coef * np.asarray(Q_sw, float).ravel() / rc
        return F_T, F_S, B_ns, B_sw

    def update(self, u: np.ndarray, v: np.ndarray, h: np.ndarray, rho: np.ndarray | None,
               ustar2_sfc: np.ndarray | None, ustar2_bot: np.ndarray, bflux: dict | None = None) -> None:
        """
        KPP on the wet columns (``kpp_column``).

        u, v : (N, ny, nx) velocities at the cell centres [m s⁻¹];  h : (N, ny, nx) layer thicknesses [m], zero
            in dry and land columns;  rho : (N, ny, nx) layer density of the model EOS [kg m⁻³];
        ustar2_sfc : (ny, nx) |τ_s|/ρ₀ [m² s⁻²] (None: 0);  ustar2_bot : unused (no bed boundary layer);
        bflux : None (no surface flux) or {"Q_ns", "Q_sw" (or None), "sw_frac_iface" (N+1, ny, nx), "fw" (or None),
            "S_sfc" (or None)} [W m⁻², m s⁻¹, g kg⁻¹].
        """
        if rho is None:
            raise ValueError("closure='kpp' needs the layer density rho")
        n = self.n
        m = self.shape[0] * self.shape[1]
        hh = np.asarray(h, float).reshape(n, m)
        wet = np.flatnonzero(hh.sum(axis=0) > 0.0)
        akv, akt, gam = np.zeros((n + 1, m)), np.zeros((n + 1, m)), np.zeros((n + 1, m))
        obl, F_T, F_S = np.zeros(m), np.zeros(m), np.zeros(m)
        if wet.size:
            hw = hh[:, wet]
            zw = np.concatenate([np.zeros((1, wet.size)), np.cumsum(hw, axis=0)])
            b = -self.g * (np.asarray(rho, float).reshape(n, m)[:, wet] - self.rho0) / self.rho0
            us = np.zeros(wet.size) if ustar2_sfc is None else np.sqrt(np.asarray(ustar2_sfc, float).ravel()[wet])
            if bflux is None:
                bns, bsw, frac = np.zeros(wet.size), None, None
            else:
                ft, fs, bns_a, bsw_a = self.buoyancy_fluxes(bflux["Q_ns"], bflux.get("Q_sw"), bflux.get("fw"),
                                                            bflux.get("S_sfc"))
                F_T[wet], F_S[wet], bns = ft[wet], fs[wet], bns_a[wet]
                bsw = None if bflux.get("Q_sw") is None else bsw_a[wet]
                frac = None if bsw is None else np.asarray(bflux["sw_frac_iface"], float).reshape(n + 1, m)[:, wet]
            f = None if self.f is None else np.asarray(self.f, float).ravel()[wet]
            r = kpp_column(hw, zw, np.asarray(u, float).reshape(n, m)[:, wet],
                           np.asarray(v, float).reshape(n, m)[:, wet], b, bns, us, frac, f, self.params,
                           bflux_sw=bsw)
            akv[:, wet], akt[:, wet], gam[:, wet], obl[wet] = r.kappa_m, r.kappa_s, r.nonlocal_, r.obl_depth
        for x in (akv, akt, gam):                                      # no flux through the bed or the surface
            x[0] = 0.0
            x[n] = 0.0
        self.akv, self.akt, self.gamma, self.obl_depth, self.F_T, self.F_S = akv, akt, gam, obl, F_T, F_S

    def tracer_diffusivity(self) -> np.ndarray:
        """κ_s (N+1, ny, nx) of the last update [m² s⁻¹] (heat, salt and passive tracers; SimpleShapes: κ_T = κ_S)."""
        return self.akt.reshape(self.n + 1, *self.shape)

    def nonlocal_flux(self) -> np.ndarray:
        """γ (N+1, ny, nx) of the last update [1]: the fraction of the non-solar surface flux carried down through
        each interface (zero at the bed and the surface)."""
        return self.gamma.reshape(self.n + 1, *self.shape)

    def nonlocal_tendency(self, h: np.ndarray, flux: np.ndarray, dt: float) -> np.ndarray:
        """
        Change of a tracer by the non-local term in one step (MOM_CVMix_KPP.F90:1899-1907, model orientation):

            Δc_k = Δt F (γ_{k+1} − γ_k) / h_k        [c]

        h : (N, ny, nx) layer thicknesses [m];  flux : (ny, nx) or (M,) kinematic surface flux F into the ocean
        [c m s⁻¹] (F_T or F_S).  γ_0 = γ_N = 0, so Σ_k h_k Δc_k = 0: the term only redistributes.  A layer with
        h_k = 0 (ALE) cannot hold its share Δt F (γ_{k+1} − γ_k); it is passed to the nearest layer with mass
        below (above at the bed), so the column content is unchanged (MOM6 divides by h + H_subroundoff instead).
        """
        n = self.n
        hh = np.asarray(h, float).reshape(n, -1)
        g = self.gamma
        q = dt * np.asarray(flux, float).ravel()[None] * (g[1:] - g[:-1])        # content change h Δc, (N, M)
        empty = hh <= 0.0
        if empty.any():
            q = q.copy()
            for k in range(n - 1, 0, -1):                             # downward: to the layer below
                mv = empty[k]
                q[k - 1] += np.where(mv, q[k], 0.0)
                q[k] = np.where(mv, 0.0, q[k])
            for k in range(n - 1):                                    # vanished bed layers: upward
                mv = empty[k] & ~empty[k + 1:].all(axis=0)
                q[k + 1] += np.where(mv, q[k], 0.0)
                q[k] = np.where(mv, 0.0, q[k])
        out = np.zeros_like(hh)
        np.divide(q, hh, out=out, where=~empty)
        return out.reshape((n,) + self.shape)

    def load(self, closure: dict | None) -> None:
        if closure is not None:
            self.akv = closure["akv"]

    def state(self) -> dict:
        return {"akv": self.akv}

    def state_fields(self) -> dict[str, np.ndarray]:
        return {STATE_PREFIX + "akv": self.akv}

    def load_state(self, d: dict[str, np.ndarray]) -> None:
        self.akv = np.asarray(d[STATE_PREFIX + "akv"]).reshape(self.n + 1, -1).copy()


def make_closure(model) -> Closure:
    """
    The closure component of a ``SigmaBaroclinic`` model, from its core settings: 'gls' → ``GLSClosure``
    (stratified with ``gls_stratified``; κ_t floor ``kv`` if > 0, else CROCO's 1e-6), anything else →
    ``ConstantClosure``.
    """
    core = model.core
    if core.closure == "kpp":
        cl = KPPClosure(core.n_layers, core.H.shape, params=getattr(model, "kpp_params", None), rho0=core.rho0,
                        g=core.g, t_coef=model.t_coef, s_coef=model.s_coef if model.salinity else 0.0, cp=model.cp)
        fq = getattr(model, "_fq", None)
        if fq is not None:                                             # f at the centres: mean of the 4 corners
            cl.f = 0.25 * (fq[:-1, :-1] + fq[:-1, 1:] + fq[1:, :-1] + fq[1:, 1:])
        return cl
    if core.closure != "gls":
        return ConstantClosure(core.closure)
    kv = float(getattr(model, "kv", 0.0))
    return GLSClosure(core.n_layers, core.H.shape, core.dt, z0=core.z0, stratified=core.gls_stratified,
                      rho0=core.rho0, g=core.g, kt_min=kv if kv > 0 else KT_MIN,
                      production=getattr(model, "gls_production", "croco"))
