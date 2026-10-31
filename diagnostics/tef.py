"""
Total exchange flow (TEF) on a transect during the run, and the mixing relations of Lorenz et al. (2021) (N2a task 7,
spec docs/superpowers/specs/2026-10-28-n2a-salinity-freshwater-design.md §6).

References
----------
* MacCready, P. (2011), Calculating estuarine exchange flow using isohaline coordinates, J. Phys. Oceanogr. 41,
  1116-1124, doi:10.1175/2011JPO4517.1 [verificar: from memory, not in the local library] — transports sorted in
  salinity bins; inflow/outflow by the sign of q(s) = −∂Q/∂s in each bin.
* Lorenz, M., K. Klingbeil and H. Burchard (2021), Impact of evaporation and precipitation on estuarine mixing,
  J. Phys. Oceanogr. 51, 1319-1333, doi:10.1175/JPO-D-20-0158.1 (local library, ``Lorenz_2021 - Impact of
  Evaporation and Precipitation on Estuarine Mixing.md``, l. 330-680).  GETM computes TEF "during the model run for
  each baroclinic time step" with 250 bins from 0 to 40 g kg⁻¹; we do the same with the face fluxes that updated the
  cells, so that relation (32) is a discrete identity.
* Lorenz, M., K. Klingbeil, P. MacCready and H. Burchard (2019), Numerical issues of the Total Exchange Flow (TEF)
  analysis framework for quantifying estuarine circulation, Ocean Sci. 15, 601-614, doi:10.5194/os-15-601-2019
  (cited by Lorenz 2021 for the extreme-value "dividing salinity" method; not used here: the in/out split is the
  sign of q(s) per bin) [verificar: from memory, not in the local library; reading_list.md].

Equations of Lorenz et al. (2021) (transcribed from the local copy; ⟨·⟩ time mean, u_n positive into the estuary at
the transect, outward at the river section A_r; E − P > 0 removes water):

  (20) Q^c(S) = ⟨∫_{A(S)} c u_n dA⟩, A(S) the part of the transect with s > S (MacCready 2011);
       Q_in^c ≥ 0 and Q_out^c ≤ 0 from Q^c(S); c = 1, s, s² give Q_in, Q_out, Q^s_in, …
  (21) c_in = Q_in^c / Q_in,   c_out = Q_out^c / Q_out
  (22) Q_r = −⟨∫_{A_r} u_n dA⟩                                 river discharge (> 0)
  (23) Q_surf = −⟨∫_{A_surf} (E − P) dA⟩ = Q_evap + Q_precip    (Q_evap ≤ 0, Q_precip ≥ 0)
  (24) F_surf^{s²} = ⟨∫_{A_surf} s² (E − P) dA⟩                 surface s² transport (a source under evaporation)
  (25) M = ⟨∫_V χ^s dV⟩ ≥ 0                                     volume-integrated salt mixing
  (26) dV  = ⟨∂_t V⟩ = Q(0) + Q_r + Q_surf = Q_in + Q_out + Q_r + Q_surf
  (27) dS  = ⟨∂_t ∫ s dV⟩ = Q^s(0) = Q_in s_in + Q_out s_out
  (28) dS2 = ⟨∂_t ∫ s² dV⟩ = Q^{s²}(0) + F_surf^{s²} − M = Q_in (s²)_in + Q_out (s²)_out + F_surf^{s²} − M
  (30) Q_in  =  s_out/(s_in − s_out) (Q_r − dV + Q_surf) + dS/(s_in − s_out)
  (31) Q_out = −s_in /(s_in − s_out) (Q_r − dV + Q_surf) − dS/(s_in − s_out)   (Knudsen 1900 with Q_surf)
  (32) M_e  = [s_out (s²)_in − s_in (s²)_out]/(s_in − s_out) (Q_r − dV + Q_surf)
              + [(s²)_in − (s²)_out]/(s_in − s_out) dS − dS2 + F_surf^{s²}                   exact
  (33) M_c  = s_in s_out (Q_r − dV + Q_surf) + (s_in + s_out) dS − dS2 + F_surf^{s²}       constancy, (s²)_x = s_x²
  (34) M_p  = [s_out (s²)_in − s_in (s²)_out]/(s_in − s_out) (Q_r + Q_surf) + F_surf^{s²}    periodicity (no storage)
  (35) M_cp = s_in s_out (Q_r + Q_surf) + F_surf^{s²}                                         both
  (36) s_surf = [(s²)_surf]^½ = (−F_surf^{s²} / Q_surf)^½
  (43) per surface cell, (S_b − S_a)/Δt = −(s_surf)² Q_surf / (1 + Δt Q_surf / V_surf)   — the model's ``S_surf``
       (task 5/6), whose region sum is F_surf^{s²}.

(30)–(31) are written here with their signs re-derived from (26)–(27) (the minus signs of (31) are lost in the text
copy).  Model signs: ``fw`` = P − E, so Q_surf = Σ fw A; the river enters at a 'clamped' edge with S = 0, so it
carries volume but no s or s².

The identity M_e = M_model holds to round-off when every flux of s and s² into the region is in the TEF transports or
the surface term: the region must hold no sponge cells (``S_sponge``), its open faces other than the transect must
carry fresh water only (rivers, S = 0) and no along-layer diffusion may cross the transect (kh = 0; the diffusive salt
flux is not an exchange-flow transport and has no volume, so TEF cannot sort it).
"""
from __future__ import annotations

import numpy as np


class TEFTransect:
    """
    Salinity-binned transports through one u-face column ``face_i`` (0 … nx) of the model, accumulated every step.

    ``sign`` = +1: flow in +x enters the region (estuary) east of the transect; −1: flow in −x enters it.  Bins:
    ``n_bins`` equal intervals of ``s_range`` [g kg⁻¹]; salinities outside go to the end bins (they still count, so
    the bin sums are the exact transect totals).
    """

    def __init__(self, face_i: int, sign: int = +1, n_bins: int = 250, s_range=(0.0, 40.0)) -> None:
        if sign not in (1, -1):
            raise ValueError(f"TEFTransect sign must be +1 or −1, got {sign!r}")
        self.face_i, self.sign, self.n_bins = int(face_i), int(sign), int(n_bins)
        self.edges = np.linspace(float(s_range[0]), float(s_range[1]), self.n_bins + 1)
        self.vol = np.zeros(self.n_bins)          # Σ Δt Q per bin [m³]
        self.salt = np.zeros(self.n_bins)         # Σ Δt F_s [g kg⁻¹ m³]
        self.salt2 = np.zeros(self.n_bins)        # Σ Δt F_{s²} [(g kg⁻¹)² m³]
        self.time = 0.0

    def add(self, Q: np.ndarray, Fs: np.ndarray, Fs2: np.ndarray, dt: float) -> None:
        """
        One step: volume flux Q [m³ s⁻¹], salt flux F_s and s² flux F_{s²} of every face-layer (N, ny) of the
        transect, in the model's +x sign.  The salinity of a face-layer is F_s/Q (MacCready 2011, the transported
        salinity); a face-layer with Q = 0 goes to bin 0 (its fluxes are 0).
        """
        Q, Fs, Fs2 = (self.sign * np.asarray(a, float).ravel() for a in (Q, Fs, Fs2))
        nz = Q != 0.0
        s = np.where(nz, Fs / np.where(nz, Q, 1.0), self.edges[0])
        ds = (self.edges[-1] - self.edges[0]) / self.n_bins
        b = np.clip(np.floor((s - self.edges[0]) / ds).astype(np.int64), 0, self.n_bins - 1)
        np.add.at(self.vol, b, dt * Q)
        np.add.at(self.salt, b, dt * Fs)
        np.add.at(self.salt2, b, dt * Fs2)
        self.time += float(dt)

    def state(self) -> dict[str, np.ndarray]:
        """Accumulators for a bit-for-bit restart: vol, salt, salt2 [per bin] and time (N2b task 5)."""
        return dict(vol=self.vol.copy(), salt=self.salt.copy(), salt2=self.salt2.copy(),
                    time=np.array(self.time, np.float64))

    def load(self, d: dict[str, np.ndarray]) -> None:
        """Restore ``state()``; the bins must be those of this transect."""
        if np.shape(d["vol"]) != self.vol.shape:
            raise ValueError(f"restart TEF has {np.shape(d['vol'])[0]} bins, the transect {self.n_bins}")
        self.vol, self.salt, self.salt2 = (np.array(d[k], np.float64) for k in ("vol", "salt", "salt2"))
        self.time = float(d["time"])

    @property
    def q(self) -> np.ndarray:
        """Time-mean volume transport per bin q(s) Δs [m³ s⁻¹] (+ into the region)."""
        return self.vol / self.time if self.time > 0 else self.vol.copy()

    def Q_of_S(self) -> np.ndarray:
        """Q(S) = transport of water saltier than S at the bin edges, Q(S) = Σ_{bins above S} q (MacCready 2011)."""
        return np.concatenate([np.cumsum(self.q[::-1])[::-1], [0.0]])

    def bulk(self) -> dict:
        """
        Time-mean bulk values (Lorenz 2021 eqs. 20–22): Q_in ≥ 0, Q_out ≤ 0 [m³ s⁻¹], Qs_in, Qs_out, Qs2_in,
        Qs2_out, s_in, s_out, s2_in = (s²)_in, s2_out = (s²)_out and Q0 = Q(0) = Q_in + Q_out.  A bin is inflow when
        its q > 0 (MacCready 2011); every other bin is outflow, so in + out are the exact transect totals.
        """
        T = self.time if self.time > 0 else 1.0
        q, qs, qs2 = self.vol / T, self.salt / T, self.salt2 / T
        inn = q > 0.0
        out = ~inn
        Q_in, Q_out = q[inn].sum(), q[out].sum()
        r = dict(Q_in=Q_in, Q_out=Q_out, Qs_in=qs[inn].sum(), Qs_out=qs[out].sum(), Qs2_in=qs2[inn].sum(),
                 Qs2_out=qs2[out].sum(), Q0=Q_in + Q_out)
        div = lambda a, b: a / b if b != 0 else np.nan            # noqa: E731
        r.update(s_in=div(r["Qs_in"], Q_in), s_out=div(r["Qs_out"], Q_out),
                 s2_in=div(r["Qs2_in"], Q_in), s2_out=div(r["Qs2_out"], Q_out))
        return r


class TEFRegion:
    """
    Region integrals of the variance budget (``SigmaState.budget``, task 6) and of the region's volume, salt and s²
    content, accumulated every step: the model side of Lorenz's relation.  ``mask`` (ny, nx): the region's cells.
    """

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = np.asarray(mask, bool)
        self.sums: dict[str, float] = {}           # Σ_t Δt Σ_region key   [c² m³]
        self.Qsurf = 0.0                           # Σ_t Δt Σ_region fw A  [m³]
        self.Qopen = 0.0                           # Σ_t Δt (inflow through the region's open faces) [m³]
        self.time = 0.0
        self.start: tuple[float, float, float] | None = None
        self.end: tuple[float, float, float] | None = None

    def state(self) -> dict[str, np.ndarray]:
        """Accumulators for a bit-for-bit restart (N2b task 5): the budget sums (one entry ``sum_<key>`` each),
        Qsurf, Qopen, time and the start/end contents (NaN when not yet set; ``has_start`` flags the start)."""
        nan = np.full(3, np.nan)
        d = {f"sum_{k}": np.array(v, np.float64) for k, v in self.sums.items()}
        d.update(Qsurf=np.array(self.Qsurf), Qopen=np.array(self.Qopen), time=np.array(self.time),
                 start=nan.copy() if self.start is None else np.array(self.start, np.float64),
                 end=nan.copy() if self.end is None else np.array(self.end, np.float64),
                 has_start=np.array(self.start is not None), has_end=np.array(self.end is not None))
        return d

    def load(self, d: dict[str, np.ndarray]) -> None:
        """Restore ``state()`` (floats are kept exactly)."""
        self.sums = {k[4:]: float(v) for k, v in d.items() if k.startswith("sum_")}
        self.Qsurf, self.Qopen, self.time = float(d["Qsurf"]), float(d["Qopen"]), float(d["time"])
        self.start = tuple(float(x) for x in d["start"]) if bool(d["has_start"]) else None
        self.end = tuple(float(x) for x in d["end"]) if bool(d["has_end"]) else None

    def content(self, V: np.ndarray, S: np.ndarray) -> tuple[float, float, float]:
        """(Σ V, Σ V S, Σ V S²) over the region; V, S (N, ny, nx)."""
        Vr = V[:, self.mask]
        Sr = S[:, self.mask]
        return float(Vr.sum()), float((Vr * Sr).sum()), float((Vr * Sr**2).sum())

    def add(self, budget: dict, fw_area: np.ndarray | None, Q_open: float, dt: float) -> None:
        """One step: the budget arrays [c² m³ s⁻¹], fw·A [m³ s⁻¹] (ny, nx) or None, the open-face inflow [m³ s⁻¹]."""
        for k, a in budget.items():
            self.sums[k] = self.sums.get(k, 0.0) + dt * float(a[:, self.mask].sum())
        if fw_area is not None:
            self.Qsurf += dt * float(np.asarray(fw_area)[self.mask].sum())
        self.Qopen += dt * float(Q_open)
        self.time += float(dt)

    def inputs(self) -> dict:
        """
        Time means for ``lorenz_mixing``: Qr (open-face inflow), Qsurf, dV, dS, dS2, Fs2_surf (Σ S_surf) and the
        model mixing M_phy = S_phy_v + S_phy_h, M_num = S_num_adv (+ S_num_remap), M_tot = M_phy + M_num, M_sponge.
        """
        T = self.time
        g = lambda k: self.sums.get(k, 0.0) / T                   # noqa: E731
        d = [(b - a) / T for a, b in zip(self.start, self.end)]
        M_phy, M_num = g("S_phy_v") + g("S_phy_h"), g("S_num_adv") + g("S_num_remap")
        return dict(Qr=self.Qopen / T, Qsurf=self.Qsurf / T, dV=d[0], dS=d[1], dS2=d[2], Fs2_surf=g("S_surf"),
                    M_phy=M_phy, M_num=M_num, M_tot=M_phy + M_num, M_sponge=g("S_sponge"))


def lorenz_mixing(bulk: dict, Qr: float, Qsurf: float, dV: float, dS: float, dS2: float,
                  Fs2_surf: float) -> dict:
    """
    Mixing relations of Lorenz et al. (2021): ``Me`` eq. (32) (exact), ``Mc`` (33) (constancy), ``Mp`` (34)
    (periodicity), ``Mcp`` (35) (both) [(g kg⁻¹)² m³ s⁻¹], and the bulk surface salinity ``s_surf`` (36) when
    Qsurf ≠ 0 and −F/Qsurf ≥ 0.  ``bulk`` from ``TEFTransect.bulk``; Qr, Qsurf [m³ s⁻¹]; dV [m³ s⁻¹], dS, dS2 the
    mean storage rates; Fs2_surf = F_surf^{s²} (eq. 24).
    """
    s_in, s_out, s2_in, s2_out = bulk["s_in"], bulk["s_out"], bulk["s2_in"], bulk["s2_out"]
    d = s_in - s_out
    fresh = Qr - dV + Qsurf
    Me = (s_out * s2_in - s_in * s2_out) / d * fresh + (s2_in - s2_out) / d * dS - dS2 + Fs2_surf      # (32)
    Mc = s_in * s_out * fresh + (s_in + s_out) * dS - dS2 + Fs2_surf                                  # (33)
    Mp = (s_out * s2_in - s_in * s2_out) / d * (Qr + Qsurf) + Fs2_surf                                # (34)
    Mcp = s_in * s_out * (Qr + Qsurf) + Fs2_surf                                                      # (35)
    s_surf = np.sqrt(-Fs2_surf / Qsurf) if Qsurf != 0 and -Fs2_surf / Qsurf >= 0 else np.nan          # (36)
    return dict(Me=Me, Mc=Mc, Mp=Mp, Mcp=Mcp, s_surf=s_surf)
