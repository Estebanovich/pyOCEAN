"""
River runoff as a surface volume source of the 3-D core (regional chain Task 8; reference reads
docs/design/regional_refs/mom6_psurf_runoff.md §3 and cefi_tools.md §5).

The discharge Q_r(t) [m³ s⁻¹] of each river cell (``forcing.sources.glofas.RiverInput``) becomes the freshwater
volume flux

    fw_r = Q_r / A_t   [m s⁻¹]   on the cell r, 0 elsewhere,

added to the real freshwater flux fw of the step (N2a): with bulk fluxes fw = (P − E) + fw_r, with a prescribed
``step(fw=...)`` fw + fw_r, otherwise fw_r alone.  It then follows the N2a path unchanged: a volume source of the
free surface, A (ηⁿ⁺¹ − ηⁿ)/Δt = −∇·(D ū)^θ + A fw (Huang 1993; MITgcm useRealFreshWaterFlux), the water entering
the top layer with S = 0 and the temperature of that layer (freshwater substep at fixed salt).  This is MOM6's
runoff in NWA12: ``lrunoff`` is summed into the net mass flux (``MOM_forcing_type.F90:638-648``), with
``AGGREGATE_FW_FORCING`` a net gain goes wholly into the top layer at the local temperature and with no salt
(``MOM_diabatic_aux.F90:990-996, 1094-1162``; ``USE_RIVER_HEAT_CONTENT = False``).  NWA12's ``DO_RIVERMIX``
(``RIVERMIX_DEPTH`` 5 m) only adds TKE to ePBL (``MOM_diabatic_aux.F90:1051-1072``); pyOCEAN has no ePBL and no
counterpart (class b/c, parity list of Task 11).

Time level: tⁿ + Δt/2, as the bulk fluxes, linear between the daily records; a time outside the records raises
``ValueError`` (``RiverInput.discharge``).  Cells that are dry (W/D) at a step lose their runoff with the rest of
fw there (the free-surface fw mask, N2a); river cells are wet coast cells.

No state: a pure function of time.  A restart checks the river set (count and a hash of the cells) and the time
origin.  Index order (j, i) at T points.
"""
from __future__ import annotations

import numpy as np


class Runoff:
    """
    River runoff component: ``fw(t)`` is the (ny, nx) volume flux Q/A_t [m s⁻¹] at model time t [s].

    Parameters
    ----------
    river : ``forcing.sources.glofas.RiverInput`` (cells, discharge(t), config()).
    area : (ny, nx) T-cell area A_t [m²].
    """

    def __init__(self, river, area: np.ndarray) -> None:
        self.river = river
        self.shape = np.shape(area)
        self.j, self.i = (np.asarray(c, dtype=np.int64) for c in np.asarray(river.cells).reshape(-1, 2).T)
        self.inv_area = 1.0 / np.asarray(area, float)[self.j, self.i]
        self.last: np.ndarray | None = None

    def fw(self, t: float) -> np.ndarray:
        """Runoff volume flux at model time t [m s⁻¹, positive into the ocean], kept as ``last``."""
        out = np.zeros(self.shape)
        np.add.at(out, (self.j, self.i), self.river.discharge(t) * self.inv_area)
        self.last = out
        return out

    def add(self, fw: np.ndarray | None, t: float) -> np.ndarray:
        """fw + runoff at t (fw None: the runoff alone)."""
        r = self.fw(t)
        return r if fw is None else np.asarray(fw, float) + r

    def config(self) -> dict:
        return self.river.config()


def runoff_errors(river, bathy) -> list[str]:
    """Pre-flight messages for ``runoff``: the type, the cells inside the grid and wet (H > 0), Q finite."""
    out: list[str] = []
    need = ("cells", "Q", "times", "discharge", "config")
    missing = [n for n in need if not hasattr(river, n)]
    if missing:
        return [f"runoff must be a forcing.sources.glofas.RiverInput (missing {', '.join(missing)}), "
                f"got {type(river).__name__}"]
    cells = np.asarray(river.cells)
    if cells.ndim != 2 or cells.shape[1] != 2:
        return [f"runoff.cells must be (n, 2) (j, i), got shape {cells.shape}"]
    Q = np.asarray(river.Q, float)
    if Q.shape != (np.size(river.times), cells.shape[0]):
        out.append(f"runoff.Q must be (n_times, n_cells) = ({np.size(river.times)}, {cells.shape[0]}), got {Q.shape}")
    elif not np.isfinite(Q).all():
        out.append("runoff.Q has non-finite values")
    if np.size(river.times) < 1:
        out.append("runoff has no time record")
    if bathy is not None and cells.size:
        H = np.asarray(bathy, float)
        ny, nx = H.shape
        j, i = cells[:, 0], cells[:, 1]
        bad = (j < 0) | (j >= ny) | (i < 0) | (i >= nx)
        if bad.any():
            out.append(f"runoff has {int(bad.sum())} cells outside the grid (ny, nx) = ({ny}, {nx})")
        ok = ~bad
        dry = np.zeros_like(bad)
        dry[ok] = ~(np.nan_to_num(H[j[ok], i[ok]], nan=0.0) > 0.0)
        if dry.any():
            out.append(f"runoff has {int(dry.sum())} cells on land (H ≤ 0), e.g. (j, i) = "
                       f"{tuple(int(x) for x in cells[np.argmax(dry)])}")
    return out
