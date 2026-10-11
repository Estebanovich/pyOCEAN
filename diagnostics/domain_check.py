"""
Pre-run checks of a model domain: bathymetry steepness, disconnected water,
pits, isolated cells and time-step limits, with a readable report.

Setting up a regional domain is the most frequent pain point in the forums of
MOM6, MITgcm, ROMS/CROCO, NEMO and Veros (docs/design/vision_roadmap.md §3b);
the checks catch, before the run, the problems that otherwise appear as
blow-ups or silent artefacts (e.g. the GEBCO 2021 pits in Guaymas bay that gave
a 0 cm M2 gauge in the northern Gulf of California nest).

Slope factors for terrain-following (σ) coordinates
---------------------------------------------------
    rx0 = |h₁ − h₂| / (h₁ + h₂)                        (Beckmann & Haidvogel 1993) [verificar]
    rx1 = max_k |z₁,ₖ − z₂,ₖ + z₁,ₖ₋₁ − z₂,ₖ₋₁| / |z₁,ₖ + z₂,ₖ − z₁,ₖ₋₁ − z₂,ₖ₋₁|   (Haney 1991) [verificar]

for adjacent cells 1, 2 and layer interfaces z_k.  Large values mean large
pressure-gradient errors (Cushman-Roisin & Beckers 2011, §20.6: the two terms of
the σ-coordinate pressure gradient must cancel for a z-only density field).
Usual limits: rx0 ≲ 0.2, rx1 ≲ 3–7 with the Shchepetkin & McWilliams (2003)
pressure gradient.

Time-step limits (explicit terms)
---------------------------------
    advective      u_max Δt / Δx_min                          (≤ 1, target ``cfl_target``)
    external wave  √(g H_max) Δt √(1/Δx² + 1/Δy²)             (only for an explicit free surface)
    internal wave  √(g′H) Δt / Δx_min                          (if g′H is given)
    inertial       f Δt                                        (≲ 1 for an accurate Coriolis term)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy import ndimage

G = 9.81


@dataclass
class WaterBody:
    """A connected set of wet cells that is not the main water body."""
    n_cells: int
    max_depth: float
    j: int
    i: int


@dataclass
class Pit:
    """A wet cell much deeper than the median of its wet neighbours."""
    j: int
    i: int
    depth: float
    neighbour_median: float


def _face_pairs(a: np.ndarray):
    """(left, right, axis-name) pairs of adjacent cells along x and y."""
    return ((a[..., :, :-1], a[..., :, 1:], "x"), (a[..., :-1, :], a[..., 1:, :], "y"))


def rx0(H: np.ndarray) -> tuple[np.ndarray, tuple[int, int, str]]:
    """
    Beckmann–Haidvogel slope factor on all wet faces (both cells with H > 0).
    Returns (values of all faces, (j, i, axis) of the maximum; i/j = first cell).
    """
    best, loc, vals = -1.0, (0, 0, "x"), []
    for lo, hi, ax in _face_pairs(np.asarray(H, float)):
        ok = np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok, np.abs(lo - hi) / (lo + hi), 0.0)
        vals.append(r.ravel())
        if r.size and r.max() > best:
            best = r.max()
            j, i = np.unravel_index(np.argmax(r), r.shape)
            loc = (int(j), int(i), ax)
    return np.concatenate(vals), loc


def rx1(H: np.ndarray, sigma_w: np.ndarray, eta: np.ndarray | None = None) -> tuple[np.ndarray, tuple[int, int, str]]:
    """
    Haney number for σ-layers with interfaces at z = σ (H + η) + η, σ ∈ [−1, 0]
    (``sigma_w`` from the bed to the surface).  Returns (values, location of max).
    """
    H = np.asarray(H, float)
    e = np.zeros_like(H) if eta is None else np.asarray(eta, float)
    z = sigma_w[:, None, None] * (H + e)[None] + e[None]
    best, loc, vals = -1.0, (0, 0, "x"), []
    for lo, hi, ax in _face_pairs(z):
        okc = _face_pairs(np.isfinite(H) & (H > 0))[0 if ax == "x" else 1]
        ok = okc[0] & okc[1]
        num = np.abs(hi[1:] - lo[1:] + hi[:-1] - lo[:-1])
        den = np.abs(hi[1:] + lo[1:] - hi[:-1] - lo[:-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(ok[None] & (den > 0), num / np.where(den > 0, den, 1.0), 0.0).max(axis=0)
        vals.append(r.ravel())
        if r.size and r.max() > best:
            best = r.max()
            j, i = np.unravel_index(np.argmax(r), r.shape)
            loc = (int(j), int(i), ax)
    return np.concatenate(vals), loc


def isolated_water_bodies(H: np.ndarray, min_depth: float = 0.0) -> list[WaterBody]:
    """
    Connected bodies of permanently wet cells (H > ``min_depth``, 4-connectivity)
    other than the largest one, largest first.  Such bodies only exchange water
    through floodable land or not at all, so tides and boundary forcing never
    reach them (e.g. a GEBCO pit behind an intertidal flat).
    """
    wet = np.isfinite(H) & (np.nan_to_num(H, nan=-np.inf) > min_depth)
    lab, n = ndimage.label(wet)
    if n <= 1:
        return []
    sizes = np.bincount(lab.ravel())[1:]
    main = int(np.argmax(sizes)) + 1
    out = []
    for k in range(1, n + 1):
        if k == main:
            continue
        cells = lab == k
        d = np.where(cells, H, -np.inf)
        j, i = np.unravel_index(np.argmax(d), d.shape)
        out.append(WaterBody(int(cells.sum()), float(H[j, i]), int(j), int(i)))
    return sorted(out, key=lambda b: -b.n_cells)


def find_pits(H: np.ndarray, ratio: float = 3.0, min_excess: float = 20.0) -> list[Pit]:
    """
    Wet cells deeper than ``ratio`` × the median of their wet 8-neighbours and by
    more than ``min_excess`` metres (data artefacts or unresolved holes).
    """
    H = np.asarray(H, float)
    wet = np.isfinite(H) & (np.nan_to_num(H) > 0)
    Hp = np.pad(np.where(wet, H, np.nan), 1, constant_values=np.nan)
    ny, nx = H.shape
    neigh = np.stack([Hp[1 + dj:1 + dj + ny, 1 + di:1 + di + nx]
                      for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(neigh, axis=0)
    flag = wet & np.isfinite(med) & (H > ratio * med) & (H - med > min_excess)
    return [Pit(int(j), int(i), float(H[j, i]), float(med[j, i])) for j, i in zip(*np.nonzero(flag))]


def isolated_cells(H: np.ndarray, min_neighbours: int = 2) -> list[tuple[int, int]]:
    """Wet cells with fewer than ``min_neighbours`` wet lateral neighbours (HYCOM topo_zthin logic)."""
    wet = (np.isfinite(H) & (np.nan_to_num(H) > 0)).astype(np.int8)
    p = np.pad(wet, 1)
    n = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return [(int(j), int(i)) for j, i in zip(*np.nonzero(wet.astype(bool) & (n < min_neighbours)))]


@dataclass
class DomainReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    dt_max_advective: float = np.inf
    cfl_target: float = 0.5

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = ["Domain check: " + ("OK" if self.ok else f"{len(self.errors)} error(s)")]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  WARNING {w}" for w in self.warnings]
        lines += [f"  info    {i}" for i in self.info]
        return "\n".join(lines)


def check_domain(h_grid: xr.Dataset, H: np.ndarray, dt: float, u_max: float = 1.0,
                 sigma_layers: int | None = None, implicit_free_surface: bool = True,
                 gprime_h: float | None = None, cfl_target: float = 0.5, rx0_max: float = 0.2,
                 rx1_max: float = 6.0, g: float = G) -> DomainReport:
    """
    Run all checks on bathymetry ``H`` [m; > 0 water, ≤ 0 floodable land, NaN wall].

    u_max : expected maximum current [m s⁻¹];  gprime_h : g′H for the internal-wave CFL [m² s⁻²]
    sigma_layers : number of σ-layers (enables rx1);  implicit_free_surface : skips the external CFL
    """
    H = np.asarray(H, float)
    rep = DomainReport(cfl_target=cfl_target)
    dx = float(np.nanmin(h_grid["dx_t"].values))
    dy = float(np.nanmin(h_grid["dy_t"].values))
    dmin = min(dx, dy)
    wet = np.isfinite(H) & (np.nan_to_num(H) > 0)
    rep.info.append(f"grid {H.shape[1]}×{H.shape[0]}, Δx_min {dmin:.0f} m, wet cells {int(wet.sum())}, "
                    f"depth {np.nanmin(np.where(wet, H, np.nan)):.1f}–{np.nanmax(np.where(wet, H, np.nan)):.1f} m")

    # time step
    c_adv = u_max * dt / dmin
    rep.dt_max_advective = dmin / u_max * cfl_target
    msg = f"advective CFL {c_adv:.2f} (u_max {u_max} m/s); Δt ≤ {rep.dt_max_advective:.0f} s for CFL {cfl_target}"
    (rep.errors if c_adv > 1 else rep.warnings if c_adv > cfl_target else rep.info).append(msg)
    if not implicit_free_surface:
        c = np.sqrt(g * np.nanmax(np.where(wet, H, np.nan))) * dt * np.sqrt(1 / dx**2 + 1 / dy**2)
        msg = f"external gravity wave CFL {c:.2f} (explicit free surface)"
        (rep.errors if c > 1 else rep.info).append(msg)
    if gprime_h is not None:
        ci = np.sqrt(gprime_h) * dt / dmin
        msg = f"internal wave CFL {ci:.2f} (√g′H = {np.sqrt(gprime_h):.2f} m/s)"
        (rep.errors if ci > 1 else rep.warnings if ci > cfl_target else rep.info).append(msg)
    fmax = float(np.nanmax(np.abs(h_grid["f_t"].values))) if "f_t" in h_grid else 0.0
    if fmax * dt > 1:
        rep.warnings.append(f"inertial f·Δt = {fmax * dt:.2f} > 1: Coriolis poorly resolved")

    # bathymetry
    r0, (j, i, ax) = rx0(H)
    msg = f"rx0 max {r0.max():.2f} at (j={j}, i={i}, {ax}-face); {int((r0 > rx0_max).sum())} faces > {rx0_max}"
    (rep.warnings if r0.max() > rx0_max else rep.info).append(msg)
    if sigma_layers:
        r1, (j, i, ax) = rx1(H, np.linspace(-1.0, 0.0, sigma_layers + 1))
        msg = f"rx1 max {r1.max():.1f} at (j={j}, i={i}, {ax}-face); {int((r1 > rx1_max).sum())} faces > {rx1_max}"
        (rep.warnings if r1.max() > rx1_max else rep.info).append(msg)
    bodies = isolated_water_bodies(H)
    msg = f"Isolated water bodies (not connected to the main one through permanently wet cells): {len(bodies)}"
    if bodies:
        b = bodies[0]
        rep.warnings.append(msg + f"; largest {b.n_cells} cells, max depth {b.max_depth:.0f} m at (j={b.j}, i={b.i})")
    else:
        rep.info.append(msg)
    pits = find_pits(H)
    if pits:
        p = max(pits, key=lambda q: q.depth - q.neighbour_median)
        rep.warnings.append(f"{len(pits)} pit(s); deepest excess {p.depth:.0f} m vs neighbours "
                            f"{p.neighbour_median:.0f} m at (j={p.j}, i={p.i})")
    iso = isolated_cells(H)
    if iso:
        rep.warnings.append(f"{len(iso)} isolated cell(s) (< 2 wet neighbours), e.g. (j={iso[0][0]}, i={iso[0][1]})")
    return rep
