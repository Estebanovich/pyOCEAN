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

import copy

import numba
import numpy as np

from core.model import threads
from core.model.contract import BuildContext, Issue, Role
from core.vertical import remap as rm
from physics.mixing.bbl_mixing import BBLParams, lotw_bbl_columns
from physics.mixing.gls import KT_MIN, N2_FLOOR, RM, RN, RP, GLSKEpsilon, buoyancy_frequency2
from physics.mixing.gls_merged import merged_columns
from physics.mixing.kappa_shear import JacksonParams, kappa_shear_columns
from physics.mixing.kpp_cvmix import KPPParams, kpp_column

STATE_PREFIX = "closure_"
H_MASSLESS = 1e-3          # [m] ALE: floor of pyOCEAN's GLS thin-layer threshold (the value of HYCOM's onemm)
THIN_FRACTION = 0.01       # [-] ALE: thin = thinner than this fraction of the column's mean layer D/N (N3 task 9c)
H_THIN_MAX = 0.1           # [m] ALE: cap of the threshold (the value of HYCOM's tencm)
THIN_RTOL = 1e-9           # [-] a layer within this relative tolerance of the threshold is thin (N3 task 9c fix 1)
PRODUCTIONS = ("croco", "energy")
SHEAR_SCHEMES = ("jackson",)
BBL_SCHEMES = ("lotw",)
KAPPA0_MIN = 1e-7          # [m² s⁻¹] floor of Jackson's κ0: KD_KAPPA_SHEAR_0 = max(KD, 1e-7), MOM_kappa_shear.F90:2170-2173


def thin_layer_threshold(h: np.ndarray, h_massless: float = H_MASSLESS, fraction: float = THIN_FRACTION,
                         h_max: float = H_THIN_MAX) -> np.ndarray:
    """
    Thin-layer threshold of the ALE GLS column (N3 task 9c): a layer with h < h_thin is merged for the closure.

        h_thin = min(h_max, max(h_massless, fraction · D/N)) · (1 + THIN_RTOL),   D = Σ_k h_k,

    with the defaults 1 mm ≤ 1 % of the mean layer thickness ≤ 10 cm.  The rule is pyOCEAN's own construction; the
    1 % is a judgement, not a calibration.  A layer is thick only for h ≥ h_thin, so a layer *at* the threshold is
    thin, as in MOM6 (a cluster opens only for ``h_2d(i,k) > dz_massless``, MOM_kappa_shear.F90:262-263) and HYCOM
    (``dpmm(k).gt.onemm``, mxkprf.F90:1871-1872).  The relative tolerance THIN_RTOL makes this hold under round-off:
    layers collapsed to the regrid minimum 1 mm come out at 1 mm ± a few 1e-16 m (a shallow z* column of 50 levels
    at D = 5 m: 1.000000000000334 mm), and where the threshold sits at its 1 mm floor (D/N ≤ 10 cm) the last bit of
    the bed depth used to decide whether the collapsed stack was merged (N3 task 9c review I1:
    ν 100× too small above an unmerged stack).  Applied here, it reaches the kernel and the oracle alike.

    h : (N, …) layer thicknesses [m]; returns h_thin (…) [m].

    Why relative to D/N (the choice is pyOCEAN's; the two reference models use absolute thresholds):

    * MOM6 (git 0e9a561, ``src/parameterizations/vertical/MOM_kappa_shear.F90``) merges layers with
      ``h_2d(i,k) > dz_massless`` false into the cluster above (l. 254-268, ``KAPPA_SHEAR_ELIM_MASSLESS``) with
      ``k0dt = dt*CS%kappa_0`` and ``dz_massless = 0.1*sqrt((US%Z_to_m*GV%m_to_H)*k0dt)`` (l. 220-221), κ₀ =
      ``KD_KAPPA_SHEAR_0`` defaulting to max(KD, 1e-7 m² s⁻¹) (l. 2170-2173): the depth a background diffusivity
      reaches in one step, times 0.1.  With pyOCEAN's κ_t floor 1e-6 m² s⁻¹ that is 0.77 mm at Lorenz's Δt ≈ 60 s
      and 2.4 mm at DOME's 600 s — below the 5 mm–2 cm layers that spike (task-5 review probe: 25× ν, 100× k), so
      the Δt-aware form does not solve the problem it is wanted for here.
    * HYCOM 2.3 (``mxkprf.F90``, ``mxmyaij``) does not merge layers: it clips thicknesses with
      ``dpmm(k) = max(onemm, dp(i,j,k,m))`` (thin layers stay in the column with Δz = 1 mm) and drops only the bed
      layers below the deepest "substantial" one, ``dpmm(k).gt.onemm`` with MASSLESS_1MM (on in the shipped build,
      Make_global.csh:100), else ``dpmm(k).gt.tencm`` (l. 1866-1878; onemm = 1 mm, tencm = 10 cm,
      mod_cb_arrays.F90:847-849).  pyOCEAN borrows only the two numbers as floor and cap.
    * An absolute threshold above a few mm fails in shallow σ-like columns: Lorenz ALE-σ has 60 layers of D/N ≥
      7.5 cm, so a 10 cm threshold would merge every layer (and freeze the column).  The relative form never merges
      a uniform column (D/N ≥ fraction · D/N), scales with the vertical resolution, and is clamped between 1 mm and
      10 cm (the values of HYCOM's onemm and tencm): Lorenz ALE-σ (D ≥ 4.5 m, N = 60) 1 mm, the CROCO diurnal
      column (150 m, N = 150) 1 cm, DOME 10 km hybrid (600–3600 m, N = 25) 10 cm.  A collapsed hybrid layer sits
      at the regrid's h_min = 1 mm (``core.vertical.regrid``), so a threshold of exactly 1 mm compared without
      tolerance left such layers (1.000000000005 mm) unmerged.
    """
    h = np.asarray(h, float)
    d_mean = h.sum(axis=0) / h.shape[0]
    return np.minimum(h_max, np.maximum(h_massless, fraction * d_mean)) * (1.0 + THIN_RTOL)


def active_layers(dz: np.ndarray, h_thin: np.ndarray, h_massless: float = H_MASSLESS) -> np.ndarray:
    """
    Layers that open a merged layer of the ALE GLS column (N3 task 9c), (N, M) bool from dz (N, M) [m] and the
    per-column threshold h_thin (M,) [m]; every other layer joins the nearest open layer above it (k' ≥ k), else
    the nearest below.

    * a layer with h ≥ h_thin;
    * the lowest of the thin layers above the topmost such layer, when those top thin layers hold ≥ ``h_massless``
      together: they form their own merged surface layer instead of joining the thick layer below.  This is MOM6's
      clustering (MOM_kappa_shear.F90:256-265: the first layer always opens cluster 1, a thin layer joins the
      cluster above it, ``nzc = nzc+1`` only for ``h_2d(i,k) > dz_massless`` once ``h_lay(nzc) > 0.0``), so thin
      layers at the surface stay a cluster of their own.  It matters for the wind: a z* floor of 1 cm layers merged
      into a 25 m layer leaves the surface ν extrapolated from an interior interface (CROCO's boundary ν), the
      stress stays in the top 1 cm and accelerates it (8 m s⁻¹ in 150 steps of the ``tests/test_gls_ale.py``
      hybrid column, against 0.58 with the 1 mm threshold).  pyOCEAN departs from MOM6 for a top cluster thinner
      than ``h_massless`` (1e-12 m massless top layers): it joins the layer below, as in N3 task 5.
    Bed thin layers join the first thick layer above them in both models.
    """
    act = dz >= h_thin[None]
    n = dz.shape[0]
    any_act = act.any(axis=0)
    kt = np.where(any_act, n - 1 - np.argmax(act[::-1], axis=0), -1)        # topmost thick layer, −1 if none
    above = np.arange(n)[:, None] > kt[None]
    top = np.where(above, dz, 0.0).sum(axis=0)
    cols = np.flatnonzero(any_act & (kt < n - 1) & (top >= h_massless))
    act[kt[cols] + 1, cols] = True
    return act


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

    @property
    def reads_velocity(self) -> bool:
        """Whether ``update`` reads u, v; False lets the driver pass None instead of building the centred
        velocities (cost fix of 2026-10-31).  A stateless closure's ``update`` does nothing, so = ``stateful``."""
        return self.stateful

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
    h_massless, thin_fraction, h_thin_max : ALE only: floor [m], fraction of D/N [-] and cap [m] of the per-column
        thin-layer threshold (``thin_layer_threshold``, N3 task 9c; defaults 1 mm, 0.01, 10 cm): layers thinner
        than it are merged for the closure column (``update``).

    ALE mode (``bind_ale``, N3 task 5; spec §3 and decisions 2–4)
    -------------------------------------------------------------
    * **Thin layers** (``_update_ale``): before N², S² and the solve, every layer with h < h_thin is merged
      with the nearest layer with mass above it (the thin layers at the top form their own merged layer when they
      hold ≥ h_massless together, else join the one below; ``active_layers``, N3 task 9c), with thickness-weighted u, v
      and ρ — MOM6 ``eliminate_massless`` (MOM_kappa_shear.F90:254-307: a massless layer joins the cluster of the
      layer above it, sums of h, u·h, T·h).  The GLS is then solved on the merged column (the same
      ``GLSKEpsilon`` code as σ) and its interface values are put back on every model interface by linear
      interpolation in the fractional height within the merged layer (MOM_kappa_shear.F90:355-363, ``kf``), so an
      interface between a vanished layer and its neighbour takes the value of the merged interface it sits on.  A
      vanished layer can therefore not create an N² or shear spike (its Δz never enters), and the bed distance of
      the log-layer ψ flux is the first layer with mass.  The threshold is per column and pyOCEAN's own rule (N3
      task 9c, ``thin_layer_threshold``): h_thin = min(10 cm, max(1 mm, 1 % of D/N)), a layer at the threshold
      counting as thin (relative tolerance, fix 1).  The 1 mm and 10 cm are the values of HYCOM's onemm and tencm,
      which HYCOM uses as a thickness clip and a bed-truncation level (mxkprf.F90:1866-1878), not as a merge
      threshold; the relative part is used because MOM6's 0.1 √(κ₀ Δt) (MOM_kappa_shear.F90:220-221) is < 3 mm in
      pyOCEAN's configurations while stale-velocity spikes reach 2 cm layers (task-5 review: 25× ν).  A
      column with one merged layer has no interior interface: k, ε, ν, κ_t go to their floors.  Only the columns
      with thin layers are merged (N3 task 10a), in the compiled kernel ``physics.mixing.gls_merged.merged_columns``
      (one column per thread task; the NumPy form ``_update_ale_numpy`` is its oracle, rtol 1e-13); every other
      column is stepped on its model layers directly (no merging arithmetic: σ ≡ ALE with fixed layers, column by
      column).
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
                 production: str = "croco", h_massless: float = H_MASSLESS, thin_fraction: float = THIN_FRACTION,
                 h_thin_max: float = H_THIN_MAX) -> None:
        self.n, self.shape, self.dt = int(n_layers), tuple(shape), dt
        self.stratified, self.rho0, self.g = bool(stratified), rho0, g
        if production not in PRODUCTIONS:
            raise ValueError(f"unknown gls_production {production!r}; use one of {PRODUCTIONS}")
        self.production, self.h_massless = production, float(h_massless)
        self.thin_fraction, self.h_thin_max = float(thin_fraction), float(h_thin_max)
        if not (0.0 < self.h_massless <= self.h_thin_max and self.thin_fraction >= 0.0):
            raise ValueError(f"GLS thin-layer threshold needs 0 < h_massless ≤ h_thin_max and thin_fraction ≥ 0, got "
                             f"{h_massless!r}, {h_thin_max!r}, {thin_fraction!r}")
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
        if self.ale:
            thr = self.thin_threshold(dz)
            if not (dz >= thr).all():
                self._update_ale(uc, vc, dz, rh, us2s, us2b, work, thr)
                return
        self._step_direct(uc, vc, dz, rh, us2s, us2b, work)

    def thin_threshold(self, dz: np.ndarray) -> np.ndarray:
        """Per-column thin-layer threshold h_thin (M,) [m] of the layers ``dz`` (N, M) (``thin_layer_threshold``)."""
        return thin_layer_threshold(dz, self.h_massless, self.thin_fraction, self.h_thin_max)

    def _step_direct(self, uc: np.ndarray, vc: np.ndarray, dz: np.ndarray, rh: np.ndarray | None,
                     us2s: np.ndarray | None, us2b: np.ndarray, work: np.ndarray | None) -> None:
        """GLS step on the model columns as they are (σ, and ALE columns without thin layers)."""
        prod = None if work is None else work / (0.5 * (dz[:-1] + dz[1:]))
        if self.stratified:
            n2 = buoyancy_frequency2(rh, dz, self.rho0, self.g)
            self.gls.step(uc, vc, dz, us2b, self.dt, ustar2_sfc=us2s, n2=n2, prod=prod)
            if self.ale:
                self._n2 = _pad_interfaces(n2)
        else:
            self.gls.step(uc, vc, dz, us2b, self.dt, prod=prod)

    def _update_ale(self, uc: np.ndarray, vc: np.ndarray, dz: np.ndarray, rho: np.ndarray | None,
                    us2s: np.ndarray | None, us2b: np.ndarray, work: np.ndarray | None,
                    thr: np.ndarray | None = None) -> None:
        """
        ALE step when some columns have thin layers (N3 task 10a): the columns without thin layers take the direct
        GLS step (``_step_direct`` on that subset: the same elementwise arithmetic as a column of a σ run, so σ ≡
        ALE holds column by column), the columns with thin layers the merged step of the compiled kernel
        ``physics.mixing.gls_merged.merged_columns`` (class docstring; oracle ``_update_ale_numpy``).
        thr : (M,) thin-layer threshold [m] (default ``thin_threshold(dz)``).
        """
        gls, n = self.gls, self.n
        thr = self.thin_threshold(dz) if thr is None else thr
        thin = (dz < thr).any(axis=0)
        tc, cc = np.flatnonzero(thin), np.flatnonzero(~thin)
        tke, eps, akv = gls.tke.copy(), gls.eps.copy(), gls.akv.copy()
        akt = gls.akt.copy() if self.stratified else None
        n2_full = np.full((n + 1, dz.shape[1]), N2_FLOOR) if self.stratified else None
        last_prod = np.zeros((n - 1, dz.shape[1]))
        if cc.size:
            sub = copy.copy(gls)                                       # shallow: constants shared, state sliced
            sub.tke, sub.eps, sub.akv = gls.tke[:, cc], gls.eps[:, cc], gls.akv[:, cc]
            if self.stratified:
                sub.akt = gls.akt[:, cc]
            saved = self._n2
            self.gls = sub
            try:
                self._step_direct(uc[:, cc], vc[:, cc], dz[:, cc], None if rho is None else rho[:, cc],
                                  None if us2s is None else us2s[cc], us2b[cc],
                                  None if work is None else work[:, cc])
            finally:
                self.gls = gls
            tke[:, cc], eps[:, cc], akv[:, cc] = sub.tke, sub.eps, sub.akv
            if akt is not None:
                akt[:, cc] = sub.akt
                n2_full[:, cc] = self._n2
            last_prod[:, cc] = sub.last_prod
            self._n2 = saved
        if tc.size:
            dummy = np.zeros((1, 1))
            mt = tc.size
            o = [np.empty((n + 1, mt)) for _ in range(5)]
            o_prod = np.empty((n - 1, mt))
            numba.set_num_threads(threads._THREADS[0])
            merged_columns(np.ascontiguousarray(dz[:, tc]), np.ascontiguousarray(uc[:, tc]),
                           np.ascontiguousarray(vc[:, tc]),
                           dummy if rho is None else np.ascontiguousarray(rho[:, tc]),
                           np.zeros(mt) if us2s is None else np.ascontiguousarray(us2s[tc], dtype=float),
                           np.ascontiguousarray(us2b[tc], dtype=float),
                           dummy if work is None else np.ascontiguousarray(work[:, tc]),
                           np.ascontiguousarray(gls.tke[:, tc]), np.ascontiguousarray(gls.eps[:, tc]),
                           np.ascontiguousarray(gls.akv[:, tc]),
                           np.ascontiguousarray(gls.akt[:, tc]) if self.stratified else dummy,
                           np.ascontiguousarray(thr[tc], dtype=float), self.h_massless, float(self.dt),
                           self.stratified, work is not None, float(self.rho0),
                           float(self.g), gls.cm0, gls.tke_min, gls.psi_min, gls.eps_min, gls.nu_min, gls.kt_min,
                           gls.galp, gls.z0s_min, gls.chk, gls.z0_bed, gls.alpha_n_min, *o, o_prod)
            tke[:, tc], eps[:, tc], akv[:, tc] = o[0], o[1], o[2]
            if akt is not None:
                akt[:, tc] = o[3]
                n2_full[:, tc] = o[4]
            last_prod[:, tc] = o_prod
        gls.tke, gls.eps, gls.akv = tke, eps, akv
        if akt is not None:
            gls.akt = akt
        gls.last_prod = last_prod
        self._n2 = n2_full

    def _update_ale_numpy(self, uc: np.ndarray, vc: np.ndarray, dz: np.ndarray, rho: np.ndarray | None,
                          us2s: np.ndarray | None, us2b: np.ndarray, work: np.ndarray | None) -> None:
        """
        NumPy form of the merged step, every column through the merging (the N3 task-5 code, kept as the oracle of
        the compiled kernel, ``tests/test_gls_ale.py::test_gls_merge_numba_matches_numpy``): merge, solve the
        columns with the same number c of merged layers together, interpolate back to the model interfaces.
        """
        gls, n = self.gls, self.n
        act = active_layers(dz, self.thin_threshold(dz), self.h_massless)   # layers that open a merged layer
        cnt = act.sum(axis=0)
        k_idx = np.arange(n)[:, None]
        # owner layer of each layer: the nearest active layer above it (k' ≥ k), else the nearest below
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

    Thin layers (final-review fix I1, ``_merged``): in columns with layers thinner than the GLS threshold
    (``thin_layer_threshold``, 1 mm … 10 cm) those layers are merged into their neighbour before the column call
    and ν, κ_s, γ are interpolated back, so a vanished ALE layer cannot set the boundary-layer depth.  σ columns
    (uniform D/N) and ALE columns without thin layers go through ``kpp_column`` unchanged.

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
        self.h_massless, self.thin_fraction, self.h_thin_max = H_MASSLESS, THIN_FRACTION, H_THIN_MAX

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
            uw, vw = np.asarray(u, float).reshape(n, m)[:, wet], np.asarray(v, float).reshape(n, m)[:, wet]
            act = active_layers(hw, thin_layer_threshold(hw, self.h_massless, self.thin_fraction, self.h_thin_max),
                                self.h_massless)
            plain = act.all(axis=0) | ~act.any(axis=0)                 # no thin layer (or every layer thin, D/N ≤
                                                                       # 1 mm): the column as it is
            ka, ks, gm, ob = (np.zeros((n + 1, wet.size)), np.zeros((n + 1, wet.size)),
                              np.zeros((n + 1, wet.size)), np.zeros(wet.size))
            if plain.any():
                c = np.flatnonzero(plain)
                r = kpp_column(hw[:, c], zw[:, c], uw[:, c], vw[:, c], b[:, c], bns[c], us[c],
                               None if frac is None else frac[:, c], None if f is None else f[c], self.params,
                               bflux_sw=None if bsw is None else bsw[c])
                ka[:, c], ks[:, c], gm[:, c], ob[c] = r.kappa_m, r.kappa_s, r.nonlocal_, r.obl_depth
            if not plain.all():
                c = np.flatnonzero(~plain)
                ka[:, c], ks[:, c], gm[:, c], ob[c] = self._merged(
                    hw[:, c], act[:, c], uw[:, c], vw[:, c], b[:, c], bns[c], us[c],
                    None if frac is None else frac[:, c], None if f is None else f[c],
                    None if bsw is None else bsw[c])
            akv[:, wet], akt[:, wet], gam[:, wet], obl[wet] = ka, ks, gm, ob
        for x in (akv, akt, gam):                                      # no flux through the bed or the surface
            x[0] = 0.0
            x[n] = 0.0
        self.akv, self.akt, self.gamma, self.obl_depth, self.F_T, self.F_S = akv, akt, gam, obl, F_T, F_S

    def _merged(self, h, act, u, v, b, bns, us, frac, f, bsw):
        """
        KPP on columns with thin layers (final-review fix I1), (N, M) inputs of those columns only, ``act`` from
        ``active_layers``.  Returns ν, κ_s, γ (N+1, M) and the boundary-layer depth (M,).

        Why: CVMix takes every layer centre as a trial depth and every layer's T, S, u, v as its own value
        (``kpp_cvmix.kpp_column``), so one 1e-10 m layer of slightly denser advected water inside the mixed layer
        sets the boundary-layer depth (61 → 17.5 m and κ_s ÷ 4 in the review probe).  MOM6 does the same: it only
        inflates thin layers to ``KPP MIN_THICKNESS`` with the error carried down (``MOM_CVMix_KPP.F90:1238-1247``,
        git 0e9a561) and still uses each of them as a trial depth with its own density (1423-1437), so MOM6 does not
        give a guard to copy.  pyOCEAN reuses its own GLS rule instead (N3 task 9c, ``thin_layer_threshold`` and
        ``active_layers``, after MOM6 ``MOM_kappa_shear.F90:254-268`` clustering): every layer with h < h_thin
        joins the nearest layer with mass above it (thin top layers holding ≥ h_massless form their own), with
        thickness-weighted u, v and b (Σ h·x / Σ h, the layer-mean that conserves momentum and buoyancy content),
        the shortwave fraction at the bottom interface of each merged layer, and KPP is computed on that merged
        column.  ν, κ_s and γ on the model interfaces are interpolated in the fractional height inside their
        merged layer, as the GLS merge does (MOM_kappa_shear.F90:355-363): the interfaces between merged layers
        take the merged value exactly, an interface inside a merged layer a value between its two bounding
        interfaces.  γ_0 = γ_N = 0 still, so the non-local term keeps conserving the column content, and its share
        in a thin layer, Δt F (γ_{k+1} − γ_k)/h_k, stays bounded (γ is linear inside the merged layer).  The bed and
        surface values of the merged column are set to zero before the interpolation (class docstring).
        """
        n, mc = h.shape
        cnt = act.sum(axis=0)
        k_idx = np.arange(n)[:, None]
        above = np.minimum.accumulate(np.where(act, k_idx, n)[::-1], axis=0)[::-1]
        below = np.maximum.accumulate(np.where(act, k_idx, -1), axis=0)
        owner = np.where(above < n, above, below)                     # the open layer each layer joins
        rank = np.cumsum(act, axis=0) - 1
        ka, ks, gm, ob = np.zeros((n + 1, mc)), np.zeros((n + 1, mc)), np.zeros((n + 1, mc)), np.zeros(mc)
        for c in np.unique(cnt):
            cols = np.flatnonzero(cnt == c)
            mg = cols.size
            lay = np.take_along_axis(rank[:, cols], np.clip(owner[:, cols], 0, n - 1), axis=0)
            colix = np.broadcast_to(np.arange(mg), (n, mg))
            h_s = h[:, cols]

            def merge(x: np.ndarray) -> np.ndarray:
                out = np.zeros((c, mg))
                np.add.at(out, (lay, colix), x)
                return out
            hc = merge(h_s)
            zf = np.concatenate([np.zeros((1, mg)), np.cumsum(h_s, axis=0)])
            zc = np.concatenate([np.zeros((1, mg)), np.cumsum(hc, axis=0)])
            kc = np.concatenate([lay, lay[-1:]])                       # merged layer of each model interface
            kf = np.clip((zf - np.take_along_axis(zc, kc, axis=0)) / np.take_along_axis(hc, kc, axis=0), 0.0, 1.0)
            kf[0], kf[-1] = 0.0, 1.0
            first = np.zeros((c + 1, mg), int)                         # model interface of each merged interface
            first[1:c] = np.cumsum(merge(np.ones((n, mg))), axis=0)[:-1].astype(int)
            first[c] = n

            def scatter(xc: np.ndarray) -> np.ndarray:
                xc = xc.copy()
                xc[0] = 0.0
                xc[c] = 0.0
                lo = np.take_along_axis(xc, kc, axis=0)
                hi = np.take_along_axis(xc, np.minimum(kc + 1, c), axis=0)
                return np.where(kf == 0.0, lo, np.where(kf == 1.0, hi, (1.0 - kf) * lo + kf * hi))
            r = kpp_column(hc, zc, merge(h_s * u[:, cols]) / hc, merge(h_s * v[:, cols]) / hc,
                           merge(h_s * b[:, cols]) / hc, bns[cols], us[cols],
                           None if frac is None else np.take_along_axis(frac[:, cols], first, axis=0),
                           None if f is None else f[cols], self.params, bflux_sw=None if bsw is None else bsw[cols])
            ka[:, cols], ks[:, cols], gm[:, cols] = scatter(r.kappa_m), scatter(r.kappa_s), scatter(r.nonlocal_)
            ob[cols] = r.obl_depth
        return ka, ks, gm, ob

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


class ShearAddedClosure(Closure):
    """
    Interior shear mixing of Jackson, Hallberg & Legg (2008) **added** to a base closure ('constant', 'none' or
    'kpp'), as MOM6 sums its vertical mixing terms (``docs/design/shear_refs/pyocean_current.md`` §4–5, option (b);
    spec 2026-10-30-jackson-shear-mixing "Decisiones tomadas por Claude").

        κ = κ_base + κ_shear,      ν = ν_base + ν_shear,      ν_shear = Pr κ_shear

    κ_base is the base's tracer diffusivity, or the constant ``kv`` when the base has none ('constant', 'none':
    the value the driver uses without shear mixing).  ν_base is the base's face viscosity, or the constant
    ``nu_const`` at every face interface when the base has none ('constant'; 'none': 0) — the ν that
    ``SigmaWetDry.column_step`` builds itself.  MOM6: Kd = Kd_bg + Kd_shear (MOM_set_diffusivity.F90:571-581),
    Kv_shear added to the viscous Kv (MOM_vert_friction.F90:2163-2205), KPP additive (MOM_CVMix_KPP.F90:419-422).
    GLS and parabolic are refused by the pre-flight: the GLS is itself a shear closure (double counting).

    Parameters
    ----------
    base : the base closure component (``ConstantClosure`` or ``KPPClosure``)
    n_layers : N;  shape : (ny, nx) of the tracer grid
    dt : model time step [s] — Jackson's Δt (pyOCEAN has no separate DT_THERM)
    f2 : (ny, nx) f² at the cell centres [s⁻²], the mean of the four corner f² (MOM_kappa_shear.F90:317-318)
    kv : constant tracer diffusivity of the model [m² s⁻¹] (κ_base of a base without κ; pushed by the model when
        ``kv`` changes after construction)
    nu_const : constant viscosity of the model [m² s⁻¹] ('constant' base; 'none': ν_base = 0)
    nu_source : object whose ``nu_const`` is read live (the model's ``SigmaWetDry`` core), so a ``nu_const`` changed
        after construction reaches ν_base as it reaches the column solver; None: the fixed ``nu_const``
    prandtl : turbulent Prandtl number Pr = ν_shear/κ_shear [-] (Jackson et al. 2008: 1; MOM6 PRANDTL_TURB)
    params : ``physics.mixing.kappa_shear.JacksonParams`` (None: MOM6 defaults, JHL set 1)
    rho0, g : ρ₀ [kg m⁻³], g [m s⁻²] of the buoyancy b = −g (ρ − ρ₀)/ρ₀

    Place in the step: ``update_phase`` 'tracers' (phase 7, as KPP and as MOM6's diabatic step): κ_shear is computed
    from the state of the step (velocities after the implicit viscosity averaged to the centres, the advected
    layer density and the layer thicknesses of the driver's ``_kpp_update``; σ: D Δσ, ALE: h) and diffuses this
    step's tracers; ν_shear is used by the implicit viscosity of the **next** step (lagged, as MOM6 Kv_shear).
    ν_shear is therefore the only state carried between steps (``SigmaState.closure['shear_kv']``, restart
    ``shear_kv`` after the base's fields); κ_shear is reseeded every call (KD_SEED_KAPPA_SHEAR).  ALE: ν_shear is
    kept by interface index through the regrid (MOM6 REMAP_AUXILIARY_VARS = False); the base remaps its own state.

    κ0 = max(background, 1e-7) m² s⁻¹ (MOM6 KD_KAPPA_SHEAR_0 = max(KD, 1e-7)) with the background ``kv`` for
    'constant'/'none' and ``KPPParams.kappa_bg`` (0 by default: κ0 = 1e-7) for KPP.  Thin layers: the per-column
    threshold of the GLS (``thin_layer_threshold``) with MOM6's merge (in ``kappa_shear_columns``).

    Columns: only wet columns (h summing to > 0; the driver passes h = 0 in dry and land columns) are computed;
    elsewhere κ = ν = 0, and the driver's ``_dry_still`` keeps their tracers still.  The bed and surface interface
    values are 0 (no flux through the bed or the surface).  Without a layer density (``rho`` None) b = 0 is not
    defined meaningfully, so κ_shear = 0 (the driver always passes ρ in phase 7).

    Attributes not defined here (``obl_depth``, ``akv``, ``gamma`` … of a KPP base) are read from the base, so
    helpers that inspect ``model._closure`` keep working with shear mixing on.

    A κ_shear (ν_shear) that is zero everywhere adds nothing: ``tracer_diffusivity`` (``viscosity``) then returns
    the base's own value unchanged (None included), so a column without shear runs bit for bit as the base.  With a
    'none' base (no ν: no implicit solve) or a constant base with kv = 0 (no tracer solve) this shortcut switches
    the whole domain between no solve and the implicit solve whenever κ_shear becomes zero or non-zero anywhere;
    the two differ only at round-off level, and the switch is what keeps zero shear = base bit for bit.
    """
    stateful = True
    update_phase = "tracers"

    def __init__(self, base: Closure, n_layers: int, shape: tuple[int, int], dt: float, f2: np.ndarray,
                 kv: float = 0.0, nu_const: float = 0.0, prandtl: float = 1.0, params: JacksonParams | None = None,
                 rho0: float = 1000.0, g: float = 9.81, nu_source=None) -> None:
        self.base = base
        self.n, self.shape, self.dt = int(n_layers), tuple(shape), float(dt)
        self.f2 = np.asarray(f2, float).reshape(-1)
        self.kv, self.prandtl = float(kv), float(prandtl)
        self._nu_const, self._nu_source = float(nu_const), nu_source
        self.params = JacksonParams() if params is None else params
        self.rho0, self.g = float(rho0), float(g)
        m = self.shape[0] * self.shape[1]
        self.kd = np.zeros((self.n + 1, m))                            # κ_shear of the last update [m² s⁻¹]
        self.kv_shear_flat = np.zeros((self.n + 1, m))                 # ν_shear (lagged state) [m² s⁻¹]
        self._per = None

    # ------------------------------------------------------------------ delegation
    def __getattr__(self, name: str):
        # only called for attributes not found normally; ``base`` itself may be missing during unpickling/copy
        if name == "base" or name.startswith("__"):
            raise AttributeError(name)
        try:
            base = self.__dict__["base"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(base, name)

    @property
    def nu_const(self) -> float:
        """ν_base of a constant base [m² s⁻¹]: the live ``nu_source.nu_const`` when given."""
        return float(self._nu_source.nu_const) if self._nu_source is not None else self._nu_const

    @property
    def restart_prefix(self) -> str:
        """File prefix of the base (``kpp_`` or none); ``shear_kv`` follows it."""
        return self.base.restart_prefix

    @property
    def per(self):
        return self._per

    @per.setter
    def per(self, value) -> None:
        self._per = value
        self.base.per = value

    @property
    def F_T(self) -> np.ndarray:
        """The base's non-solar kinematic heat flux of the non-local term (KPP), else zeros (no non-local term)."""
        f = getattr(self.base, "F_T", None)
        return np.zeros(self.kd.shape[1]) if f is None else f

    @property
    def F_S(self) -> np.ndarray:
        f = getattr(self.base, "F_S", None)
        return np.zeros(self.kd.shape[1]) if f is None else f

    def nonlocal_tendency(self, h: np.ndarray, flux: np.ndarray, dt: float) -> np.ndarray:
        return self.base.nonlocal_tendency(h, flux, dt)

    def nonlocal_flux(self) -> np.ndarray | None:
        return self.base.nonlocal_flux()

    def bind_ale(self, remap_code: int) -> None:
        self.base.bind_ale(remap_code)

    def remap(self, h_src: np.ndarray, h_dst: np.ndarray) -> None:
        """The base remaps its own state; ν_shear stays by interface index (MOM6 REMAP_AUXILIARY_VARS = False)."""
        self.base.remap(h_src, h_dst)

    def check(self, ctx: BuildContext) -> list[Issue]:
        return self.base.check(ctx)

    # ------------------------------------------------------------------ fields
    @property
    def kd_shear(self) -> np.ndarray:
        """κ_shear (N+1, ny, nx) of the last update [m² s⁻¹] (registry ``kd_shear``)."""
        return self.kd.reshape(self.n + 1, *self.shape)

    @property
    def kv_shear(self) -> np.ndarray:
        """ν_shear = Pr κ_shear (N+1, ny, nx) at the cell centres [m² s⁻¹] (registry ``kv_shear``), lagged."""
        return self.kv_shear_flat.reshape(self.n + 1, *self.shape)

    def kappa0(self) -> float:
        """Jackson's background κ0 [m² s⁻¹] = max(background κ, 1e-7) (class docstring)."""
        bg = self.base.params.kappa_bg if isinstance(self.base, KPPClosure) else self.kv
        return max(float(bg), KAPPA0_MIN)

    # ------------------------------------------------------------------ interface
    def viscosity(self, st) -> tuple[np.ndarray, np.ndarray] | None:
        """ν_base + ν_shear at the u- and v-face interfaces (``centre_to_faces``, periodic wrap faces with ``per``);
        the base's own value when ν_shear is zero everywhere."""
        if st is not None:
            self.load(getattr(st, "closure", None))
        nb = self.base.viscosity(None)
        if not self.kv_shear_flat.any():
            return nb
        su, sv = centre_to_faces(self.kv_shear, self._per)
        if nb is None:
            nu0 = self.nu_const if getattr(self.base, "kind", "constant") == "constant" else 0.0
            return nu0 + su, nu0 + sv
        return nb[0] + su, nb[1] + sv

    def update(self, u: np.ndarray, v: np.ndarray, h: np.ndarray, rho: np.ndarray | None,
               ustar2_sfc: np.ndarray | None, ustar2_bot: np.ndarray, bflux: dict | None = None) -> None:
        """
        The base's update, then κ_shear of the wet columns (``kappa_shear_columns``) and ν_shear = Pr κ_shear.

        u, v : (N, ny, nx) velocities at the cell centres [m s⁻¹];  h : (N, ny, nx) layer thicknesses [m], zero in
        dry and land columns;  rho : (N, ny, nx) layer density [kg m⁻³] (None: κ_shear = 0);  the rest goes to
        the base.
        """
        self.base.update(u, v, h, rho, ustar2_sfc, ustar2_bot, bflux=bflux)
        n = self.n
        m = self.kd.shape[1]
        hh = np.asarray(h, float).reshape(n, m)
        wet = np.flatnonzero(hh.sum(axis=0) > 0.0)
        kd = np.zeros((n + 1, m))
        if wet.size and rho is not None:
            dz = np.ascontiguousarray(hh[:, wet])
            # buoyancy b = −g (ρ − ρ₀)/ρ₀ [m s⁻²]
            b = -self.g * (np.asarray(rho, float).reshape(n, m)[:, wet] - self.rho0) / self.rho0
            uw = np.asarray(u, float).reshape(n, m)[:, wet]
            vw = np.asarray(v, float).reshape(n, m)[:, wet]
            numba.set_num_threads(threads._THREADS[0])
            kappa, _ = kappa_shear_columns(uw, vw, b, dz, self.f2[wet], self.dt, self.kappa0(),
                                           thin_layer_threshold(dz), self.params)
            kd[:, wet] = kappa
        kd[0] = 0.0                                                    # no flux through the bed or the surface
        kd[n] = 0.0
        self.kd = kd
        self.kv_shear_flat = self.prandtl * kd

    def tracer_diffusivity(self) -> np.ndarray | None:
        """κ_base + κ_shear (N+1, ny, nx) [m² s⁻¹]; the base's own value when κ_shear is zero everywhere."""
        kb = self.base.tracer_diffusivity()
        if not self.kd.any():
            return kb
        if kb is None:
            return self.kv + self.kd_shear
        return kb + self.kd_shear

    def load(self, closure: dict | None) -> None:
        if closure is not None:
            self.base.load(closure)
            if "shear_kv" in closure:
                self.kv_shear_flat = closure["shear_kv"]

    def state(self) -> dict:
        out = dict(self.base.state() or {})
        out["shear_kv"] = self.kv_shear_flat
        return out

    def state_fields(self) -> dict[str, np.ndarray]:
        out = dict(self.base.state_fields()) if self.base.stateful else {}
        out[STATE_PREFIX + "shear_kv"] = self.kv_shear_flat
        return out

    def load_state(self, d: dict[str, np.ndarray]) -> None:
        if self.base.stateful:
            self.base.load_state(d)
        self.kv_shear_flat = np.asarray(d[STATE_PREFIX + "shear_kv"]).reshape(self.n + 1, -1).copy()


class BBLAddedClosure(Closure):
    """
    Drag-driven bottom-boundary-layer diapycnal mixing **added** to a closure ('constant', 'none', 'kpp', or any of
    them already wrapped in ``ShearAddedClosure``): the energy-limited law-of-the-wall diffusivity of MOM6
    ``add_LOTW_BBL_diffusivity`` (``physics.mixing.bbl_mixing``; spec 2026-10-30-bbl-drag-mixing-design.md, the
    OM4 scheme, MOM_set_diffusivity.F90:1606-1789), additive as in OM4 (SD:1782-1783):

        κ = κ_inner + Kd_BBL,      ν = ν_inner   (no viscosity, no Prandtl number: MOM6 neither)

    κ_inner is the wrapped closure's tracer diffusivity (Jackson's κ_base + κ_shear), or the constant ``kv`` when it
    has none ('constant', 'none' without shear: the value the driver uses without BBL mixing).

    Inputs of each update (driver phase 7, ``Driver._kpp_update``; MOM6 computes Kd_BBL in its diabatic step too):

    * E (ny, nx) [m³ s⁻³] = ``bbl_energy``: the kinetic energy per unit area and density that pyOCEAN's bed drag
      removed in this step, Σ_k r_k u_k² at the faces averaged to the centres (``Driver._ale_bed_energy``).
      **pyOCEAN's physical drag work** (≈ C_d |U|³), not MOM6's ``BBL_meanKE_loss`` (≈ 2 C_d |U|³ with ≈ √2 u*,
      task 2, class (d)): Esteban's decision of 2026-10-30, so BBL_EFFIC = 0.2 here equals 0.1 in MOM6/OM4.
    * TKE_col = ``bbl_effic`` · E (SD:1709);
    * u* = √(u*²_b), u*²_b = |τ_b|/ρ₀ of the same drag (``ctx.ustar2_bot``, U_bg and the law of the wall included);
      the velocities are those after the implicit viscosity (MOM6: start of step; class (c), spec decision 2);
    * |f| at the centres = ¼ Σ |f| of the four corners, in MOM6's order (SD:1680-1681);
    * N² from the layer density of the **merged** column (thickness-weighted merged densities over the merged
      spacing, as the ALE GLS and Jackson; = ``physics.mixing.gls.buoyancy_frequency2`` in a column without thin
      layers; final review I1), computed inside ``lotw_bbl_columns``;
    * the thin-layer threshold of the ALE closures (``thin_layer_threshold``); the merge is pyOCEAN's choice
      (``physics.mixing.bbl_mixing`` section "Columns with thin-layer merge"; MOM6's LOTW does not merge).

    Kd_BBL is computed and used in the same step: no state of its own (``state``, ``load``, ``state_fields``,
    ``load_state``, ``remap`` are those of the wrapped closure).  Only wet columns (h summing to > 0) are computed;
    elsewhere Kd_BBL = 0, and the driver's ``_dry_still`` keeps their tracers still.  A Kd_BBL that is zero
    everywhere returns the wrapped closure's own κ unchanged (None included), so no drag work = base bit for bit.

    ``update_phase`` 'tracers' and ``stateful`` True, as ``ShearAddedClosure``, so the driver calls ``update`` in
    phase 7 even when the wrapped closure is the constant one.  Attributes not defined here are read from the
    wrapped closure (``obl_depth`` of KPP, ``kd_shear``/``params`` of Jackson …); ``kv`` and ``prandtl`` set on the
    wrapper reach the wrapped closure, so the model's ``kv``/``shear_prandtl`` setters keep working.

    Parameters
    ----------
    base : the wrapped closure;  n_layers : N;  shape : (ny, nx) of the tracer grid
    absf : (ny, nx) |f| at the cell centres [s⁻¹]
    kv : constant tracer diffusivity of the model [m² s⁻¹] (κ_inner of a closure without κ)
    params : ``physics.mixing.bbl_mixing.BBLParams`` (None: MOM6/OM4 defaults)
    rho0, g : ρ₀ [kg m⁻³], g [m s⁻²] of N² = −(g/ρ₀) ∂ρ/∂z
    """
    stateful = True
    update_phase = "tracers"

    def __init__(self, base: Closure, n_layers: int, shape: tuple[int, int], absf: np.ndarray, kv: float = 0.0,
                 params: BBLParams | None = None, rho0: float = 1000.0, g: float = 9.81) -> None:
        self.base = base
        self.n, self.shape = int(n_layers), tuple(shape)
        self.absf = np.asarray(absf, float).reshape(-1)
        self._kv = float(kv)
        self.bbl_params = BBLParams() if params is None else params
        self.rho0, self.g = float(rho0), float(g)
        m = self.shape[0] * self.shape[1]
        self.kd = np.zeros((self.n + 1, m))                            # Kd_BBL of the last update [m² s⁻¹]
        self.energy_flat = np.zeros(m)                                 # drag work E of the last update [m³ s⁻³]
        self._per = None

    # ------------------------------------------------------------------ delegation
    def __getattr__(self, name: str):
        if name == "base" or name.startswith("__"):
            raise AttributeError(name)
        try:
            base = self.__dict__["base"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(base, name)

    @property
    def kv(self) -> float:
        """Constant tracer diffusivity [m² s⁻¹]; set: also the wrapped closure's (Jackson κ_base, κ0)."""
        return self._kv

    @kv.setter
    def kv(self, value: float) -> None:
        self._kv = float(value)
        if "kv" in vars(self.base):                                     # ShearAddedClosure: κ_base and κ0
            self.base.kv = float(value)

    @property
    def prandtl(self) -> float:
        """Jackson's Prandtl number of the wrapped shear closure (AttributeError without one)."""
        return self.base.prandtl

    @prandtl.setter
    def prandtl(self, value: float) -> None:
        if not hasattr(self.base, "prandtl"):
            raise AttributeError("prandtl: no shear mixing under the BBL mixing")
        self.base.prandtl = value

    @property
    def restart_prefix(self) -> str:
        return self.base.restart_prefix

    @property
    def reads_velocity(self) -> bool:
        """The BBL part reads only h, ρ, u*_b and the drag work: u, v only for the wrapped closure."""
        return self.base.reads_velocity

    @property
    def per(self):
        return self._per

    @per.setter
    def per(self, value) -> None:
        self._per = value
        self.base.per = value

    @property
    def F_T(self) -> np.ndarray:
        """The wrapped closure's non-local heat flux (KPP), else zeros (no non-local term)."""
        f = getattr(self.base, "F_T", None)
        return np.zeros(self.kd.shape[1]) if f is None else f

    @property
    def F_S(self) -> np.ndarray:
        f = getattr(self.base, "F_S", None)
        return np.zeros(self.kd.shape[1]) if f is None else f

    def nonlocal_tendency(self, h: np.ndarray, flux: np.ndarray, dt: float) -> np.ndarray:
        return self.base.nonlocal_tendency(h, flux, dt)

    def nonlocal_flux(self) -> np.ndarray | None:
        return self.base.nonlocal_flux()

    def bind_ale(self, remap_code: int) -> None:
        self.base.bind_ale(remap_code)

    def remap(self, h_src: np.ndarray, h_dst: np.ndarray) -> None:
        """The wrapped closure remaps its own state; Kd_BBL has none."""
        self.base.remap(h_src, h_dst)

    def check(self, ctx: BuildContext) -> list[Issue]:
        return self.base.check(ctx)

    # ------------------------------------------------------------------ fields
    @property
    def kd_bbl(self) -> np.ndarray:
        """Kd_BBL (N+1, ny, nx) at the interfaces of the last update [m² s⁻¹] (registry ``kd_bbl``)."""
        return self.kd.reshape(self.n + 1, *self.shape)

    @property
    def energy(self) -> np.ndarray:
        """Drag work E (ny, nx) of the last update [m³ s⁻³] (before the factor ``bbl_effic``)."""
        return self.energy_flat.reshape(self.shape)

    # ------------------------------------------------------------------ interface
    def viscosity(self, st) -> tuple[np.ndarray, np.ndarray] | None:
        """The wrapped closure's ν: the BBL mixing produces no viscosity (MOM6 SD:374, 486-488)."""
        return self.base.viscosity(st)

    def update(self, u: np.ndarray, v: np.ndarray, h: np.ndarray, rho: np.ndarray | None,
               ustar2_sfc: np.ndarray | None, ustar2_bot: np.ndarray, bflux: dict | None = None,
               bbl_energy: np.ndarray | None = None) -> None:
        """
        The wrapped closure's update, then Kd_BBL of the wet columns (``lotw_bbl_columns``).

        u, v : (N, ny, nx) velocities at the cell centres [m s⁻¹] (to the wrapped closure);  h : (N, ny, nx) layer
        thicknesses [m], zero in dry and land columns;  rho : (N, ny, nx) layer density [kg m⁻³] (None: Kd_BBL = 0);
        ustar2_bot : (ny, nx) |τ_b|/ρ₀ [m² s⁻²];  bbl_energy : (ny, nx) drag work E [m³ s⁻³] (None: 0).
        """
        self.base.update(u, v, h, rho, ustar2_sfc, ustar2_bot, bflux=bflux)
        n = self.n
        m = self.kd.shape[1]
        energy = np.zeros(m) if bbl_energy is None else np.asarray(bbl_energy, float).reshape(m)
        if rho is None:
            kd = np.zeros((n + 1, m))
        else:
            # the whole grid goes to the kernel, which skips dry and land columns (Σh ≤ h_massless) before reading
            # ρ: no gather/scatter of the wet columns (cost fix of 2026-10-31; same Kd bit for bit)
            dz = np.asarray(h, float).reshape(n, m)
            p = self.bbl_params
            ustar = np.sqrt(np.asarray(ustar2_bot, float).reshape(m))
            tke = p.bbl_effic * energy                                 # TKE_col = BBL_EFFIC · E (SD:1709)
            numba.set_num_threads(threads._THREADS[0])
            # N² from the merged densities inside (final review I1; = buoyancy_frequency2 without thin layers)
            kd = lotw_bbl_columns(dz, np.asarray(rho, float).reshape(n, m), ustar, self.absf, tke,
                                  thin_layer_threshold(dz), p, rho0=self.rho0, g=self.g)
        kd[0] = 0.0                                                    # no flux through the bed or the surface
        kd[n] = 0.0
        self.kd = kd
        self.energy_flat = energy

    def tracer_diffusivity(self) -> np.ndarray | None:
        """κ_inner + Kd_BBL (N+1, ny, nx) [m² s⁻¹]; the wrapped closure's own value when Kd_BBL is zero everywhere."""
        kb = self.base.tracer_diffusivity()
        if not self.kd.any():
            return kb
        if kb is None:
            return self.kv + self.kd_bbl
        return kb + self.kd_bbl

    def load(self, closure: dict | None) -> None:
        self.base.load(closure)

    def state(self) -> dict | None:
        return self.base.state()

    def state_fields(self) -> dict[str, np.ndarray]:
        return dict(self.base.state_fields()) if self.base.stateful else {}

    def load_state(self, d: dict[str, np.ndarray]) -> None:
        if self.base.stateful:
            self.base.load_state(d)


def find_closure(cl: Closure, kind: type) -> Closure | None:
    """The closure of class ``kind`` in the wrapper chain ``cl`` (BBL → shear → base), or None."""
    while cl is not None:
        if isinstance(cl, kind):
            return cl
        cl = cl.__dict__.get("base") if isinstance(cl, (ShearAddedClosure, BBLAddedClosure)) else None
    return None


def make_closure(model) -> Closure:
    """
    The closure component of a ``SigmaBaroclinic`` model, from its core settings: 'gls' → ``GLSClosure``
    (stratified with ``gls_stratified``; κ_t floor ``kv`` if > 0, else CROCO's 1e-6), anything else →
    ``ConstantClosure``; with ``shear_mixing='jackson'`` that closure wrapped in ``ShearAddedClosure``; with
    ``bbl_mixing='lotw'`` the result wrapped in ``BBLAddedClosure``.
    """
    core = model.core
    cl = _shear_closure(model)
    if getattr(model, "bbl_mixing", None) is None:
        return cl
    # |f| at the centres: ¼ Σ |f| of the four corners, MOM6's order (MOM_set_diffusivity.F90:1680-1681)
    fq = np.abs(np.asarray(model._fq, float))
    absf = 0.25 * ((fq[:-1, :-1] + fq[1:, 1:]) + (fq[1:, :-1] + fq[:-1, 1:]))
    return BBLAddedClosure(cl, core.n_layers, core.H.shape, absf, kv=float(getattr(model, "kv", 0.0)),
                           params=getattr(model, "bbl_params", None), rho0=core.rho0, g=core.g)


def _shear_closure(model) -> Closure:
    """The base closure, wrapped in ``ShearAddedClosure`` with ``shear_mixing='jackson'``."""
    core = model.core
    cl = _base_closure(model)
    if getattr(model, "shear_mixing", None) is None:
        return cl
    # f² at the centres: the mean of the four corner f² (MOM_kappa_shear.F90:317-318)
    fq = np.asarray(model._fq, float)
    f2 = 0.25 * (fq[:-1, :-1] ** 2 + fq[:-1, 1:] ** 2 + fq[1:, :-1] ** 2 + fq[1:, 1:] ** 2)
    return ShearAddedClosure(cl, core.n_layers, core.H.shape, core.dt, f2, kv=float(getattr(model, "kv", 0.0)),
                             nu_const=float(core.nu_const), prandtl=float(getattr(model, "shear_prandtl", 1.0)),
                             params=getattr(model, "jackson_params", None), rho0=core.rho0, g=core.g,
                             nu_source=core)


def _base_closure(model) -> Closure:
    """The closure of ``core.closure`` alone (``make_closure`` without the added shear mixing)."""
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
