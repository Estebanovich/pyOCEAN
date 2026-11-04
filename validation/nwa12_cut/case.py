"""
NWA12 cut: one case definition for pyOCEAN and MOM6 (spec docs/superpowers/specs/2026-11-01-regional-chain-design.md,
"Decisiones del caso"; plan Task 9).

The Mid-Atlantic Bight and southern New England on the NWA12 grid of CEFI (``ocean_hgrid.nc``, ``ocean_topog.nc``,
Zenodo 7893727): the T cells inside 75.5–68.5 °W, 36.5–42.0 °N (ruling R13: north limit 42.0 °N so Narragansett
Bay / Blackstone and Buzzards Bay are inside), 2017-05-01 → 2017-09-01.  Edge runs of fewer than 4 wet cells are
land in the case topography (``edit_edges``, ruling R14: upper Delaware Bay on the west edge).  Every edge with a
wet cell is open
('radiating' with the GLORYS exterior and the TPXO10 tide); the others are coast.

Parity with NWA12 (``parity_config``; values from ``docs/design/regional_refs/*.md`` and rulings R3–R5, R7, R11, R12):

| item | NWA12 (MOM6) | pyOCEAN | class of the difference |
|---|---|---|---|
| vertical | z* 75 levels ``vgrid_75_2m.nc`` | ALE z*, ``dz_top`` = the same 75 Δz | — |
| momentum | vector invariant, ``LAPLACIAN`` False, ``BIHARMONIC`` True, ``SMAGORINSKY_AH`` with ``SMAG_BI_CONST`` 0.03, ``AH_VEL_SCALE`` 0.01, ``AH`` 0, ``BOUND_AH``, ``HORVISC_BOUND_COEF`` 0.8, ``USE_LAND_MASK_FOR_HVISC``, ``OBC_ZERO_BIHARMONIC``, no ``BOUND_CORIOLIS_BIHARM``/Leith | vector invariant, ``smagorinsky_bi=0.03``, ``ah_vel_scale=0.01``, ``ah_bi=0``, Laplacian off (``smagorinsky=0``, ``ah=0``) (``core.dynamics.hor_visc.biharmonic``, Gulf Stream fix step 4, ``docs/design/regional_refs/mom6_biharmonic.md``); was Laplacian Smagorinsky 0.15 until 2026-11-03 | c (bound × 6/11 for AB3; thin-layer ratio on wet faces only; biharmonic shear stress 0 at the open-edge corners, MOM6 takes it from the halo) |
| gravity | ``G_EARTH`` 9.8 | ``g=9.8`` (R21; reaches the free surface and Flather, the PGF, the tidal potential/SAL, the closures and the EOS pressure; the FMS bulk keeps its own GRAV 9.80) | — |
| PGF bottom mass weighting | ``MASS_WEIGHT_IN_PRESSURE_GRADIENT`` True (every face) | ``pgf_mass_weight='full'`` (R21; initial PGF = MOM6 to 1.2e-14, Task 10) | — |
| EOS | ``WRIGHT_FULL`` θ, S_P, ρ₀ 1035, ``C_P`` 3992 | ``eos='wright_full'``, ``ts_kind='theta_sp'``, ρ₀ 1035, c_p 3992 | — |
| surface | FMS coupler NCAR bulk (SIS2) | ``BulkFluxes('ncar')`` (= FMS ``surface_flux`` bit for bit), ERA5 + lsm | c (no SIS2, gusts) |
| shortwave | ``VAR_PEN_SW``, ``OPACITY_SCHEME`` MANIZZA_05 with chlorophyll from file, 3 bands (``mom6_full``; the run ``mom6_full_jerlov`` is the Jerlov control) | Jerlov I, Paulson & Simpson | c |
| mixing | ePBL + Jackson + double diffusion, ``KD`` 1e-6, ``KV`` 0 (possibly a background viscosity ``KD``·``PRANDTL_BKGND`` = 1e-6, MOM_bkgnd_mixing.F90:430 [hipótesis, branch not checked with ePBL]) | KPP (CVMix) + Jackson, κ_bg 1e-6 (parity value, R16; pyOCEAN default 0), ν_bg 0 | b (no ePBL, no double diffusion) |
| shear mixing | Jackson, ``PRANDTL_TURB`` 1.25 (ν = 1.25 κ, MOM_kappa_shear.F90:393, 773) | ``shear_prandtl=1.25`` | — |
| Coriolis | ``BOUND_CORIOLIS`` False, ``CORIOLIS_EN_DIS`` True (of two transports, the one that dissipates energy, MOM_CoriolisAdv.F90:439-449, 733-815) | Sadourny energy PV flux with BOUND_CORIOLIS always on (core/components/momentum.py:210, not a parameter), no ``CORIOLIS_EN_DIS`` | c |
| mixed-layer restratification | ``MIXEDLAYER_RESTRAT`` True, Fox-Kemper et al. (2011), ``FOX_KEMPER_ML_RESTRAT_COEF`` 1.0, ``MLE_FRONT_LENGTH`` 1500 m, MLD from ePBL | none | b (found 2026-11-04, Gulf Stream drift research) |
| time scheme | split RK2, ``BE`` 0.6; split-explicit barotropic (``BEBT`` 0.2, ``DTBT`` −0.9) | AB3; implicit free surface θ = 0.6, θ_bc = ½ | c |
| drag | quadratic ``CDRAG`` 0.003, ``HBBL`` 10, ``DRAG_BG_VEL`` 0, ``CHANNEL_DRAG`` | quadratic cd 0.003, hbbl 10, U_bg 0 | b/c (no channel drag, R5) |
| open edges | Flather + Orlanski + nudging 3 d / 360 d (by Orlanski phase speed, ``OBC_RAD_VEL_WT`` 0.3), reservoir 9 km in and out; runs with ``ENABLE_BUGS_BY_DEFAULT`` True (``OBC_PROJECTION_BUG``, ``EXTERIOR_OBC_BUG``, ``OBC_TRACER_DZ_BUG``) | 'radiating': Flather + Orlanski + nudging τ_in 3 d, τ_out 360 d, chosen by the sign of the Orlanski phase speed cx as CROCO (core/components/open_boundary.py:150; R7 said "by flow direction", corrected 2026-11-04), no reservoir/sponge | c |
| tangential velocity at open edges | ``ORLANSKI_TAN``, ``NUDGED_TAN`` (radiated and nudged toward the exterior tangential velocity), ``OBC_COMPUTED_VORTICITY``/``STRAIN`` | default ``obc_tangential='none'``: not radiated or nudged, ζ = 0 and D_S = 0 at the open-edge corners (MOM6's ZERO configuration, not free slip; the open-face normal velocity enters the Smagorinsky tension in both modes since the fix of 2026-11-03); ``obc_tangential='mom6'`` ports the MOM6 treatment (``docs/design/regional_refs/mom6_obc_tangential.md``), exterior ``OBData.u_tangential`` from GLORYS at the edge corners | c ('none') / — ('mom6') |
| boundary tide | TPXO, 10 constituents, η and TPXO's velocity added to the segment, nodal fixed at a date | ``BoundaryTides`` TPXO10, 10 constituents, ``velocity_depth='tpxo'``: u = U/h_TPXO, TPXO's own velocity (R19, revises R3 for the case; u = U/D_model gave up to 3.8 m/s on 5-m faces), OTPS nodal at each time, ramp 0 | c (nodal) |
| body tide | ``TIDES``, scalar SAL 0.05, SAL in Flather | ``tidal_potential=True``, ``sal_beta`` 0.05, ``sal_flather`` | c (legacy SAL path, R4) |
| atm. pressure | off (``MAX_P_SURF`` = 0) | ``atm_pressure=False`` | — |
| rivers | GloFAS v3 (CEFI ``write_runoff_glofas.py``), ePBL ``DO_RIVERMIX`` | GloFAS v4 with the CEFI ldd rule (R11, R12), real freshwater at the top layer | c (version; no river mixing without ePBL, b) |
| Δt | ``DT`` 600 s, ``DT_THERM`` = ``DT_TRACER_ADVECT`` 1800 s | one Δt: 600 s for every phase (300 s until the θ_bc / PV-flux fix ``e1ffbbb``, 2026-11-02, R15) | c (no separate tracer Δt) |
| tracer advection | ``PPM:H3`` every 1800 s | WENO5-Z + FCT in SSP-RK3 (three stages) every 600 s | c |

Why ALE z* + vector-invariant momentum + a Smagorinsky stress-tensor operator (biharmonic as NWA12, or the Laplacian;
not σ or the constant Laplacian): the σ flux-form momentum and the constant-A_h Laplacian have no curvilinear metric
terms (Task 3, ``docs/design/regional_refs/metric_audit.md``
§2.3, class b); the invariant forms are metric-correct, as MOM6's.  ``parity_config`` enforces it and the pre-flight
warns for the other choices on a supergrid grid.

Depths as MOM6 with ``MASKING_DEPTH`` 1, ``MINIMUM_DEPTH`` 4, ``MAXIMUM_DEPTH`` 6500 m (MOM_parameter_doc.short:89-97):
land where depth ≤ 1 m, depths in (1, 4) m raised to 4 m, deeper than 6500 m cut to 6500 m.

Index order: (j, i) on the T points of the cut, (k, j, i) with k = 0 at the bed for model fields.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

REFS = Path.home() / "pyOCEAN_refs"
RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut")
DATA = RUNS / "data"

MASKING_DEPTH, MINIMUM_DEPTH, MAXIMUM_DEPTH = 1.0, 4.0, 6500.0     # [m] NWA12 MOM_parameter_doc.short:89-97
CONSTITUENTS = ("m2", "s2", "n2", "k2", "k1", "o1", "p1", "q1", "mm", "mf")   # OBC_TIDE_CONSTITUENTS
#: fixed observing points that every open edge must stay ≥ ``MIN_EDGE_CELLS`` cells away from
BUOY = {"ndbc_44025": (40.251, -73.164), "ooi_pioneer": (40.1, -70.8)}
MIN_EDGE_CELLS = 10
#: shortest run of wet edge cells kept as an open boundary (ruling R14): shorter runs are estuary or inlet interiors
#: that the 1/12° exterior (GLORYS) does not resolve; they become land in the case topography
MIN_EDGE_RUN = 4
EDGES = ("west", "east", "south", "north")


@dataclass(frozen=True)
class Case:
    """Paths, window and period of one regional cut (the real NWA12 cut is ``CASE``; tests build small ones)."""
    name: str
    hgrid: Path                                  # MOM6 supergrid ocean_hgrid.nc
    topog: Path                                  # ocean_topog.nc (depth, positive down)
    vgrid: Path                                  # vgrid file with dz (surface first) [m]
    glorys: tuple[Path, ...]                     # GLORYS12 daily subsets
    era5: Path                                   # directory of era5_*_{instant,accum}.nc
    era5_lsm: Path | None                        # ERA5 land-sea mask file
    glofas: Path                                 # directory of glofas_v4_consolidated_YYYY-MM.nc
    ldd: Path | None                             # GloFAS v4 local drain direction (CEFI rule, R11)
    tpxo: Any                                    # TPXO10 atlas directory, or an xr.Dataset of constants (tests)
    tides: Path | None                           # CO-OPS gauge files (station metadata)
    runs: Path                                   # output root
    t0: np.datetime64 = np.datetime64("2017-05-01T00:00", "ns")
    t1: np.datetime64 = np.datetime64("2017-09-01T00:00", "ns")
    box: tuple[float, float, float, float] = (-75.5, -68.5, 36.5, 42.0)   # lon0, lon1, lat0, lat1 [°E, °N]
    window: tuple[int, int, int, int] | None = None                         # (i0, i1, j0, j1) of choose_window
    era5_pattern: str = "era5_*_{kind}.nc"
    glofas_pattern: str = "glofas_v4_consolidated_*.nc"
    points: dict = field(default_factory=dict)  # extra series points {name: (lat, lon)}
    tide_ramp_days: float = 0.0                  # NWA12: RAMP_OBCS = False (R3); tests ramp
    tide_velocity_depth: str = "tpxo"            # BoundaryTides velocity_depth: TPXO's own velocity (R19, as CEFI/
                                                 # MOM6); 'model' = U/D_model (R3, the library default)

    @property
    def inputs(self) -> Path:
        return self.runs / "inputs" / "pyocean"


#: The NWA12 cut.  ``window`` is what ``choose_window`` returns on the real grid (checked by
#: tests/test_nwa12_cut.py::test_case_window_is_choose_window when the grid is on disk): 87 × 89 T cells.
CASE = Case(
    name="nwa12_cut",
    hgrid=REFS / "nwa12_grid" / "ocean_hgrid.nc",
    topog=REFS / "nwa12_grid" / "ocean_topog.nc",
    vgrid=REFS / "nwa12_grid" / "vgrid_75_2m.nc",       # = CEFI tools/grid/vgrid_75_2m.nc (identical dz)
    glorys=(DATA / "glorys" / "glorys12_my_20170429_20170903_box.nc",),
    era5=DATA / "era5",
    era5_lsm=DATA / "era5" / "era5_nwa12_cut_lsm.nc",
    glofas=DATA / "glofas",
    ldd=REFS / "glofas_aux" / "ldd_glofas_v4_0.nc",
    tpxo=REFS / "TPXO10_atlas_v2_nc",
    tides=DATA / "tides",
    runs=RUNS,
    window=(282, 369, 425, 514),
)


# ---------------------------------------------------------------------------------------------------- window
def mom6_depth(depth: np.ndarray) -> np.ndarray:
    """MOM6 depth rules of NWA12 (module docstring): NaN = land, else clipped to [MINIMUM_DEPTH, MAXIMUM_DEPTH]."""
    d = np.nan_to_num(np.asarray(depth, float), nan=0.0)
    return np.where(d > MASKING_DEPTH, np.clip(d, MINIMUM_DEPTH, MAXIMUM_DEPTH), np.nan)


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of the runs of True in a 1-D boolean array."""
    out, i, n = [], 0, row.size
    while i < n:
        if row[i]:
            j = i
            while j < n and row[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def edit_edges(H: np.ndarray, min_run: int = MIN_EDGE_RUN) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Case topography edit of ruling R14 (class c): every run of fewer than ``min_run`` contiguous wet cells along an
    edge of the window is set to land (NaN), so that no open boundary cuts across an estuary interior (e.g. the 2
    wet cells of upper Delaware Bay on the NWA12 cut's west edge; the Delaware enters as GloFAS runoff).  Removing
    edge cells exposes no new edge cell, so one pass suffices.  The edited depth is the case topography for every
    model (pyOCEAN and MOM6, Task 10).  Runs of ``min_run`` cells or more stay open even when they are not connected
    inside the window to the edge's other runs: such a run (Cape Cod Bay on the north edge, 6 cells) exchanges with
    open water outside the cut (Massachusetts Bay), and closing it would turn it into a closed basin.

    Returns (edited H, [(j, i), …] of the cells set to land).
    """
    H = np.array(H, float)
    ny, nx = H.shape
    edges = {"west": [(j, 0) for j in range(ny)], "east": [(j, nx - 1) for j in range(ny)],
             "south": [(0, i) for i in range(nx)], "north": [(ny - 1, i) for i in range(nx)]}
    cells = []
    for pts in edges.values():
        row = np.array([np.isfinite(H[p]) for p in pts])
        for a, n in _runs(row):
            if n < min_run:
                cells.extend(pts[a:a + n])
    cells = sorted(set(cells))
    for p in cells:
        H[p] = np.nan
    return H, cells


def open_edges(H: np.ndarray) -> tuple[str, ...]:
    """Edges with at least one wet cell (they are open; the others are coast)."""
    wet = np.isfinite(H)
    rows = {"west": wet[:, 0], "east": wet[:, -1], "south": wet[0, :], "north": wet[-1, :]}
    return tuple(e for e in EDGES if rows[e].any())


def _nearest_cell(lon: np.ndarray, lat: np.ndarray, la: float, lo: float) -> tuple[int, int]:
    d = (np.cos(np.deg2rad(la)) * (lon - lo)) ** 2 + (lat - la) ** 2
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return int(j), int(i)


def choose_window(lon: np.ndarray, lat: np.ndarray, depth: np.ndarray,
                  box: tuple[float, float, float, float] = CASE.box, points: dict | None = None,
                  min_cells: int = MIN_EDGE_CELLS) -> tuple[int, int, int, int]:
    """
    Index window (i0, i1, j0, j1) (end exclusive) of the T cells of the full grid (``lon``, ``lat`` (nj, ni) [°E,
    °N], ``depth`` (nj, ni) [m, positive down]) inside ``box`` = (lon0, lon1, lat0, lat1), with the spec rules:

    1. the window is the index bounding box of the T points inside the box;
    2. edge runs of fewer than ``MIN_EDGE_RUN`` wet cells are land (``edit_edges``, ruling R14); every edge with a
       wet cell left is open, and each open edge is ≥ ``min_cells`` cells from every
       point of ``points`` (default ``BUOY``: NDBC 44025 and OOI Pioneer); an edge that is too close is moved
       outward until it is not, and the open edges are re-evaluated;
    3. no open-edge cell has H ≤ 0 (the ALE open boundary does not admit floodable cells): ValueError otherwise.
    """
    points = BUOY if points is None else points
    lon0, lon1, lat0, lat1 = box
    inside = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
    if not inside.any():
        raise ValueError(f"no T point of the grid inside the box {box}")
    jj, ii = np.nonzero(inside)
    i0, i1, j0, j1 = int(ii.min()), int(ii.max()) + 1, int(jj.min()), int(jj.max()) + 1
    nj, ni = lon.shape
    cells = {name: _nearest_cell(lon, lat, la, lo) for name, (la, lo) in points.items()}
    for _ in range(4 * (ni + nj)):
        H = edit_edges(mom6_depth(depth[j0:j1, i0:i1]))[0]
        moved = False
        for e in open_edges(H):
            for name, (jc, ic) in cells.items():
                if not (j0 <= jc < j1 and i0 <= ic < i1):
                    raise ValueError(f"point {name} is outside the window {(i0, i1, j0, j1)}")
                dist = {"west": ic - i0, "east": i1 - 1 - ic, "south": jc - j0, "north": j1 - 1 - jc}[e]
                if dist < min_cells:
                    step = min_cells - dist
                    lim = {"west": i0 == 0, "east": i1 == ni, "south": j0 == 0, "north": j1 == nj}[e]
                    if lim:
                        raise ValueError(f"open edge {e} is {dist} cells from {name} (< {min_cells}) and cannot "
                                         "move outward: it is the edge of the parent grid")
                    if e == "west":
                        i0 = max(i0 - step, 0)
                    elif e == "east":
                        i1 = min(i1 + step, ni)
                    elif e == "south":
                        j0 = max(j0 - step, 0)
                    else:
                        j1 = min(j1 + step, nj)
                    moved = True
        if not moved:
            break
    else:
        raise ValueError("choose_window did not converge")
    H = edit_edges(mom6_depth(depth[j0:j1, i0:i1]))[0]
    for e, row in (("west", H[:, 0]), ("east", H[:, -1]), ("south", H[0, :]), ("north", H[-1, :])):
        if e in open_edges(H) and (np.isfinite(row) & (row <= 0.0)).any():
            raise ValueError(f"open edge {e} has cells with H ≤ 0")
    return i0, i1, j0, j1


# ---------------------------------------------------------------------------------------------------- inputs
def load_hgrid(case: Case) -> xr.Dataset:
    with xr.open_dataset(case.hgrid) as hg:
        return hg.load()


def load_depth(case: Case) -> np.ndarray:
    with xr.open_dataset(case.topog) as t:
        return np.asarray(t["depth"].values, float)


def window_of(case: Case) -> tuple[int, int, int, int]:
    """The case window, chosen by ``choose_window`` when the case does not fix it."""
    if case.window is not None:
        return case.window
    hg = load_hgrid(case)
    return choose_window(hg["x"].values[1::2, 1::2], hg["y"].values[1::2, 1::2], load_depth(case), case.box,
                         case.points or None)


def grid(case: Case) -> tuple[xr.Dataset, np.ndarray]:
    """(horizontal grid of the cut, H (ny, nx) [m] with NaN on land), MOM6 metrics (core.grid.supergrid); H is the
    case topography: MOM6 depth rules, then the edge edit of ruling R14 (``edit_edges``)."""
    from core.grid.supergrid import grid_from_supergrid
    i0, i1, j0, j1 = window_of(case)
    H, cells = edit_edges(mom6_depth(load_depth(case)[j0:j1, i0:i1]))
    hg = grid_from_supergrid(load_hgrid(case), depth=np.nan_to_num(H), window=(i0, i1, j0, j1))
    hg.attrs["topography_edits"] = ";".join(f"{j},{i}" for j, i in cells)     # (j, i) of the cut set to land, R14
    return hg, H


def topography_edits(hg: xr.Dataset) -> list[dict]:
    """The R14 edits recorded by ``grid``: [{'j', 'i', 'lat', 'lon'}] in the cut's indices."""
    txt = hg.attrs.get("topography_edits", "")
    out = []
    for item in filter(None, txt.split(";")):
        j, i = (int(x) for x in item.split(","))
        out.append(dict(j=j, i=i, lat=float(hg["lat_t"].values[j, i]), lon=float(hg["lon_t"].values[j, i])))
    return out


def vgrid_dz(case: Case) -> np.ndarray:
    """z* nominal thicknesses, surface first [m] (ALE_COORDINATE_CONFIG = FILE:vgrid_75_2m.nc,dz)."""
    with xr.open_dataset(case.vgrid) as v:
        return np.asarray(v["dz"].values, float)


def _files(directory: Path, pattern: str) -> list[Path]:
    return sorted(Path(directory).glob(pattern))


def era5_files(case: Case) -> list[Path]:
    """ERA5 files of the months (or other chunks) that have both their instant and accum file (a download in
    progress may hold one of the two)."""
    inst = {f.name.replace("instant", "{kind}"): f for f in _files(case.era5, case.era5_pattern.format(kind="instant"))}
    acc = {f.name.replace("accum", "{kind}"): f for f in _files(case.era5, case.era5_pattern.format(kind="accum"))}
    both = sorted(set(inst) & set(acc))
    return [inst[k] for k in both] + [acc[k] for k in both]


def era5_coverage(case: Case) -> np.datetime64:
    """Last hourly record of the complete ERA5 chunks (``era5_files``) [datetime64]; ``case.t0`` without files."""
    fs = era5_files(case)
    if not fs:
        return case.t0
    ends = []
    for f in fs:
        with xr.open_dataset(f) as d:
            tn = "valid_time" if "valid_time" in d.coords else "time"
            ends.append(d[tn].values.max())
    return np.datetime64(min(ends[: len(fs) // 2][-1:] + ends[len(fs) // 2:][-1:]), "ns")


def stations(case: Case) -> dict[str, tuple[float, float]]:
    """Series points {name: (lat, lon)}: the CO-OPS gauges (metadata of their water-level files), the buoys, extras."""
    out = {}
    if case.tides is not None and Path(case.tides).exists():
        for f in sorted(Path(case.tides).glob("wl_*_20170501.json")):
            md = json.loads(f.read_text())["metadata"]
            out[f"coops_{md['id']}"] = (float(md["lat"]), float(md["lon"]))
    out.update(BUOY)
    out.update(case.points)
    return out


def station_cells(hg: xr.Dataset, H: np.ndarray, pts: dict) -> dict[str, tuple[int, int]]:
    """Nearest wet T cell of each point (j, i); points with no wet cell within 3 cells are dropped."""
    lon, lat = hg["lon_t"].values, hg["lat_t"].values
    wet = np.isfinite(H)
    out = {}
    for name, (la, lo) in pts.items():
        d = (np.cos(np.deg2rad(la)) * (lon - lo)) ** 2 + (lat - la) ** 2
        dd = np.where(wet, d, np.inf)
        j, i = np.unravel_index(np.argmin(dd), dd.shape)
        if np.sqrt(dd[j, i]) <= 3.0 * float(np.abs(np.diff(lat[:, 0])).max()):
            out[name] = (int(j), int(i))
    return out


def tide_constants(case: Case, hg: xr.Dataset) -> xr.Dataset:
    """TPXO10 constants (h, U, V) of the 10 NWA12 constituents over the cut + 0.5° (or the case's Dataset)."""
    if isinstance(case.tpxo, xr.Dataset):
        return case.tpxo
    from forcing.sources.tpxo import read_tpxo, tpxo_atlas_files
    h, u, g, names = tpxo_atlas_files(case.tpxo, CONSTITUENTS)
    if tuple(names) != CONSTITUENTS:
        raise ValueError(f"TPXO atlas {case.tpxo} lacks constituents {set(CONSTITUENTS) - set(names)}")
    lon, lat = hg["lon_t"].values, hg["lat_t"].values
    return read_tpxo(h, u, g, lon_range=(float(lon.min()) - 0.5, float(lon.max()) + 0.5),
                     lat_range=(float(lat.min()) - 0.5, float(lat.max()) + 0.5))


# ---------------------------------------------------------------------------------------------------- ocean
def glorys_ocean(case: Case, hg: xr.Dataset, t0: np.datetime64, t1: np.datetime64) -> xr.Dataset:
    """Canonical GLORYS (θ, S_P, u, v east/north, zos) covering the cut + 0.2°, records bracketing [t0, t1]."""
    from forcing.sources.base import BBox
    from forcing.sources.glorys import GLORYSSource
    return GLORYSSource(list(case.glorys)).load(_dt(t0), _dt(t1), BBox.from_grid(hg, pad=0.2))


def _dt(t: np.datetime64) -> datetime:
    return datetime.fromisoformat(str(np.datetime64(t, "s")))


def _edge_strip(hg: xr.Dataset, side: str) -> xr.Dataset:
    """The boundary row/column of T points and the two face rows/columns around it (``exterior_provider``)."""
    sl = {"west": dict(i_t=slice(0, 1), i_u=slice(0, 2)), "east": dict(i_t=slice(-1, None), i_u=slice(-2, None)),
          "south": dict(j_t=slice(0, 1), j_v=slice(0, 2)), "north": dict(j_t=slice(-1, None), j_v=slice(-2, None))}
    return hg.isel(**sl[side])


def exterior(case: Case, hg: xr.Dataset, H: np.ndarray, edges: tuple[str, ...], ocean: xr.Dataset,
             t0: np.datetime64, t1: np.datetime64):
    """GLORYS exterior t → {edge: OBData} on the data's own levels (MOM6 segment data, CEFI ``z_to_dz``)."""
    from forcing.ocean_init import exterior_provider, prepare_ocean, source_level_grid
    vsrc = source_level_grid(ocean["depth"].values)
    # q_points: the tangential velocity at the edge corners (OBData.u_tangential, NUDGED_TAN target of
    # obc_tangential='mom6'; ignored otherwise)
    prep = {e: prepare_ocean(ocean, _edge_strip(hg, e), vsrc, q_points=True) for e in edges}
    depth = {"west": H[:, 0], "east": H[:, -1], "south": H[0, :], "north": H[-1, :]}
    return exterior_provider(prep, vsrc, {e: depth[e] for e in edges}, case.t0, t0, t1, remove_mean_ssh=False)


def initial_ocean(case: Case, m, hg: xr.Dataset, H: np.ndarray, ocean: xr.Dataset, t0: np.datetime64):
    """
    Initial state at t0 from GLORYS (MOM6 NWA12: T, S, SSH and velocities from file, Z_INIT_ALE_REMAPPING):
    θ, S_P remapped from the GLORYS levels (``source_layers`` → ``initial_state(h_src=...)``, content exact per
    column), η = zos (product datum kept, as MOM6), u, v rotated to the grid and interpolated linearly in depth to
    the initial layer centres of each face (MOM6 remaps them; class c), 0 in layers thinner than 1 cm.
    """
    from forcing.interp import interp_time
    from forcing.ocean_init import prepare_ocean, prepare_ocean_source, source_layers, source_level_grid
    times = ocean["time"].values
    k = int(np.clip(np.searchsorted(times, np.datetime64(t0, "ns"), side="right") - 1, 0, times.size - 1))
    ocean = ocean.isel(time=slice(k, min(k + 2, times.size)))         # the records bracketing t0
    src = prepare_ocean_source(ocean, hg)
    vsrc = source_level_grid(ocean["depth"].values)
    prep = prepare_ocean(ocean, hg, vsrc)
    eta = np.where(np.isfinite(H), interp_time(prep["t"][["ssh"]], t0)["ssh"].values, 0.0)
    h_src, T_src, S_src = source_layers(src, np.nan_to_num(H), eta, t0)
    st = m.initial_state(eta, T_src, h_src=h_src, S=S_src, ts_kind="theta_sp")
    hu, hv = m.face_thickness(st.h)
    d = -np.asarray(vsrc["z_t"].values, float)
    for name, h_face, attr in (("u", hu, "u"), ("v", hv, "v")):
        f = interp_time(prep[name][[name]], t0)[name].values                  # (n_src, ny_f, nx_f), surface first
        bottom = np.cumsum(h_face[::-1], axis=0)[::-1]                       # depth below η of each layer's bed
        zc = bottom - 0.5 * h_face                                           # layer-centre depth [m]
        vel = _interp_depth(d, f, zc)
        setattr(st, attr, np.where(h_face > 0.01, vel, 0.0))
    return st


def _interp_depth(d: np.ndarray, f: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Linear interpolation in depth of columns f (n_src, …) at depths d (n_src,) to depths z (N, …), constant
    beyond the ends; NaN source values are filled downward with the last valid value first."""
    f = np.asarray(f, float).copy()
    for k in range(1, f.shape[0]):
        f[k] = np.where(np.isnan(f[k]), f[k - 1], f[k])
    f = np.nan_to_num(f)
    zc = np.clip(z, d[0], d[-1])
    k = np.clip(np.searchsorted(d, zc, side="right") - 1, 0, d.size - 2)
    w = (zc - d[k]) / (d[k + 1] - d[k])
    flat_f = f.reshape(f.shape[0], -1)
    kk = k.reshape(k.shape[0], -1)
    cols = np.arange(flat_f.shape[1])[None, :]
    a, b = flat_f[kk, cols], flat_f[kk + 1, cols]
    return ((1.0 - w.reshape(kk.shape)) * a + w.reshape(kk.shape) * b).reshape(z.shape)


# ---------------------------------------------------------------------------------------------------- model
def parity_config(case: Case, hg: xr.Dataset, H: np.ndarray, dt: float, *, edges: tuple[str, ...],
                  boundary_tides=None, surface_fluxes=None, runoff=None, open_boundary_state=None,
                  **overrides) -> dict:
    """
    Keyword arguments of ``SigmaBaroclinic`` for the NWA12 parity configuration (module docstring table).  The
    forcing objects are built by ``build``; ``overrides`` replace single entries (tests) but the coordinate, the
    momentum form and a Smagorinsky stress-tensor viscosity (biharmonic, NWA12's, or Laplacian) are enforced
    (curvilinear grid, Task 3 limitation).
    """
    from physics.mixing.kpp_cvmix import KPPParams
    dz = vgrid_dz(case)
    cfg = dict(
        dt=float(dt), n_layers=dz.size, vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=dz),
        momentum="vector_invariant", pgf="fv", g=9.8, pgf_mass_weight="full",
        # NWA12 LAPLACIAN False, BIHARMONIC True, SMAG_BI_CONST 0.03, AH_VEL_SCALE 0.01, AH 0 (Gulf Stream fix step 4)
        smagorinsky=0.0, ah=0.0, smagorinsky_bi=0.03, ah_vel_scale=0.01, ah_bi=0.0,
        eos="wright_full", ts_kind="theta_sp", rho0=None, cp=3992.0, salinity=True,
        closure="kpp", kpp_params=KPPParams(kappa_bg=1e-6, nu_bg=0.0), shear_mixing="jackson",
        shear_prandtl=1.25,
        z0=None, bottom_drag="quadratic", cd=0.003, hbbl=10.0, drag_bg_vel=0.0,
        open_boundaries=tuple(edges), open_boundary_mode={e: "radiating" for e in edges},
        open_boundary_state=open_boundary_state, obc_tau_in=3.0 * 86400.0, obc_tau_out=360.0 * 86400.0,
        obc_tangential="mom6",  # NWA12 ORLANSKI_TAN, NUDGED_TAN, computed ζ/D_S (ruling R27, 2026-11-03)
        boundary_tides=boundary_tides, tidal_potential=boundary_tides is not None, sal_beta=0.05,
        sal_flather=boundary_tides is not None, atm_pressure=False,
        surface_fluxes=surface_fluxes, runoff=runoff, mix_diagnostic=False,
    )
    cfg.update(overrides)
    stress_tensor = cfg["smagorinsky"] > 0 or cfg.get("smagorinsky_bi", 0.0) > 0
    if cfg["vertical_coordinate"] != "ale" or cfg["momentum"] != "vector_invariant" or not stress_tensor:
        raise ValueError("the NWA12 cut needs vertical_coordinate='ale', momentum='vector_invariant' and "
                         "smagorinsky_bi > 0 (or smagorinsky > 0): the σ flux form and the constant Laplacian lack "
                         "curvilinear metric terms (metric_audit.md §2.3)")
    return cfg


@dataclass
class Built:
    """A built run: model, initial state and what the runner records."""
    model: Any
    state: Any
    hg: xr.Dataset
    H: np.ndarray
    edges: tuple[str, ...]
    cells: dict
    river: Any = None


def build(case: Case, mode: str, dt: float, days: float, hg: xr.Dataset | None = None, H: np.ndarray | None = None,
          river=None, **overrides) -> Built:
    """
    The model and initial state of ``mode``:
      'tides' — homogeneous ocean (θ = 10 °C, S_P = 35) at rest, boundary tides + tidal potential + SAL only (no
                GLORYS, atmosphere or rivers), the parity physics otherwise;
      'full'  — the parity configuration from t0 for ``days`` days: GLORYS initial state and exterior, ERA5 bulk,
                GloFAS rivers, tides.
    """
    from core.components.boundary_tides import BoundaryTides
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    if hg is None or H is None:
        hg, H = grid(case)
    edges = open_edges(H)
    t0 = case.t0
    t1 = t0 + np.timedelta64(int(round(days * 86400)), "s")
    bt = BoundaryTides(tide_constants(case, hg), list(CONSTITUENTS), ref_date=t0, ramp_days=case.tide_ramp_days,
                       velocity_depth=case.tide_velocity_depth)
    cells = station_cells(hg, H, stations(case))
    if mode == "tides":
        cfg = parity_config(case, hg, H, dt, edges=edges, boundary_tides=bt, **overrides)
        m = SigmaBaroclinic(hg, H, **cfg)
        T = np.full(H.shape, 10.0)
        st = m.initial_state(np.zeros(H.shape), np.broadcast_to(T, (m.n,) + H.shape).copy(),
                             S=np.full((m.n,) + H.shape, 35.0))
        return Built(m, st, hg, H, edges, cells)
    if mode != "full":
        raise ValueError(f"unknown mode {mode!r}; use 'tides' or 'full'")
    from core.components.surface_fluxes import BulkFluxes
    from forcing.atmosphere import era5_atmosphere
    ocean = glorys_ocean(case, hg, t0, t1)
    obs = exterior(case, hg, H, edges, ocean, t0, t1)
    atm = era5_atmosphere(era5_files(case), hg, t0, _dt(t0), _dt(t1),
                          lsm=None if case.era5_lsm is None else case.era5_lsm)
    if river is None:
        river = rivers(case, hg, H)
    cfg = parity_config(case, hg, H, dt, edges=edges, boundary_tides=bt, surface_fluxes=BulkFluxes(atm, "ncar"),
                        runoff=river, open_boundary_state=obs, **overrides)
    m = SigmaBaroclinic(hg, H, **cfg)
    st = initial_ocean(case, m, hg, H, ocean, t0)
    return Built(m, st, hg, H, edges, cells, river)


def rivers(case: Case, hg: xr.Dataset, H: np.ndarray):
    """
    GloFAS v4 river mouths on the cut's coast cells, CEFI ldd rule (R11, R12), origin = case.t0.  The pour points
    are selected (mean discharge ≥ 1 m³ s⁻¹) over the **whole case period** ``case.t0 → case.t1``, not over a run's
    own window (ruling R17): every run and every restart of the case has the same river set (the restart checks it),
    whatever its ``--days``.
    """
    from forcing.sources.glofas import glofas_mouths, open_ldd
    files = _files(case.glofas, case.glofas_pattern)
    ds = xr.open_mfdataset(files, combine="by_coords") if len(files) > 1 else xr.open_dataset(files[0])
    try:
        ldd = None if case.ldd is None else open_ldd(case.ldd)
        return glofas_mouths(ds.load(), hg, case.t0, case.t1, origin=case.t0, ldd=ldd, wet=np.isfinite(H))
    finally:
        ds.close()


def deep_sponge(case: Case, hg: xr.Dataset, H: np.ndarray, tau_days: float = 5.0, h_min: float = 500.0,
                taper_cells: int = 3, period: tuple[str, str] = ("2017-05-01", "2017-06-30"),
                t_ref_depth: float | None = 200.0, t_ref_max: float = 15.0,
                per_column: bool = False) -> tuple[dict, dict]:
    """
    Diagnostic interior T/S sponge over deep water (intrusion diagnosis, ``figs/intrusion_diag/README.md``; not a
    parity item).  Returns (``SigmaBaroclinic`` keyword arguments, summary for ``meta.json``).

    The interior sponge relaxes T, S implicitly (``core.components.sponge``, T ← (T + Δt r T*)/(1 + Δt r), MOM6
    MOM_sponge.F90 implicit form).

    ``per_column=True`` (``--deep-sponge-3d``, ruling 2026-11-04): the target of each model column is the GLORYS
    time mean over ``period`` at its nearest GLORYS point, on the GLORYS levels (interfaces ``source_level_grid``)
    down to the deepest level any sponge column needs; values below the GLORYS bed take the value above (MOM6
    MOM_ALE_sponge.F90:1006-1014), and a column with no GLORYS value takes that of the nearest column with one
    (``_fill_target``).  ``t_ref_depth`` is not used.  Otherwise one profile: the profile is the area-weighted horizontal mean of the GLORYS12 time mean over ``period`` (θ [°C],
    S_P), sampled at the nearest GLORYS point of each model T cell, over the wet cells with H > ``h_min`` and, when
    ``t_ref_depth`` is given, GLORYS θ(``t_ref_depth``) < ``t_ref_max`` (slope water north of the Gulf Stream north
    wall, the 15 °C at 200 m criterion), on the GLORYS levels with any value (interfaces ``source_level_grid``).

    Rate r = w / (τ · 86400 s) [s⁻¹] on the wet cells with H > ``h_min``, 0 elsewhere; w = min(1, d / taper_cells)
    with d the Euclidean distance in cells to the nearest wet cell with H ≤ ``h_min`` (w = 1 for taper_cells = 0).
    """
    from forcing.ocean_init import source_level_grid
    lat, lon, area = hg["lat_t"].values, hg["lon_t"].values, hg["area_t"].values
    wet = np.isfinite(H)
    deep = wet & (np.nan_to_num(H) > h_min)
    with xr.open_dataset(case.glorys[0]) as ds:
        ds = ds.sel(time=slice(period[0], period[1]))
        if ds.sizes["time"] == 0:
            raise ValueError(f"no GLORYS records in {period}")
        T = ds["thetao"].mean("time").values.astype(float)
        S = ds["so"].mean("time").values.astype(float)
        d = ds["depth"].values.astype(float)
        jj = np.abs(ds["latitude"].values[:, None, None] - lat[None]).argmin(0)
        ii = np.abs(ds["longitude"].values[:, None, None] - lon[None]).argmin(0)
        n_rec = int(ds.sizes["time"])
    Tm, Sm = T[:, jj, ii], S[:, jj, ii]
    if per_column:
        return _deep_sponge_columns(Tm, Sm, d, H, wet, deep, tau_days, h_min, taper_cells, period, n_rec)
    sel = deep.copy()
    if t_ref_depth is not None:
        k = int(np.argmin(np.abs(d - t_ref_depth)))
        sel &= np.nan_to_num(Tm[k], nan=np.inf) < t_ref_max
    w = np.where(sel[None] & np.isfinite(Tm) & np.isfinite(Sm), area[None], 0.0)
    wk = w.sum(axis=(1, 2))
    ok = wk > 0
    if not ok.any():
        raise ValueError("no deep cells selected for the sponge profile")
    T_prof = (np.nan_to_num(Tm) * w).sum(axis=(1, 2))[ok] / wk[ok]
    S_prof = (np.nan_to_num(Sm) * w).sum(axis=(1, 2))[ok] / wk[ok]
    zw = source_level_grid(d[ok])["z_f"].values
    rate = _deep_sponge_rate(wet, deep, tau_days, taper_cells)
    kw = dict(sponge_rate=rate, sponge_T=T_prof, sponge_zw=zw, sponge_S=S_prof)
    summary = dict(tau_days=tau_days, h_min=h_min, taper_cells=taper_cells, period=list(period), records=n_rec,
                   t_ref_depth=t_ref_depth, t_ref_max=t_ref_max, profile_cells=int(sel.sum()),
                   sponge_cells=int(deep.sum()), levels=int(ok.sum()),
                   T_surface=float(T_prof[0]), S_surface=float(S_prof[0]))
    return kw, summary


def _deep_sponge_rate(wet: np.ndarray, deep: np.ndarray, tau_days: float, taper_cells: int) -> np.ndarray:
    """r = w / (τ · 86400 s) [s⁻¹] on ``deep``, w = min(1, d / taper_cells), d the distance in cells to the nearest
    wet shallow cell (``deep_sponge``)."""
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~(wet & ~deep))
    wt = np.ones(deep.shape) if taper_cells <= 0 else np.minimum(1.0, dist / float(taper_cells))
    return np.where(deep, wt / (tau_days * 86400.0), 0.0)


def _fill_target(a: np.ndarray) -> np.ndarray:
    """
    A GLORYS field (nz, ny, nx) sampled on the model cells, made finite: NaN below the GLORYS bed takes the value
    of the level above (MOM6 MOM_ALE_sponge.F90:1006-1014, ``tmpT1d(k) = tmpT1d(k-1)``); a column with no value at
    all takes the column of the nearest cell (Euclidean, in cells) that has one.
    """
    from scipy.ndimage import distance_transform_edt
    out = np.array(a, float)
    for k in range(1, out.shape[0]):
        bad = ~np.isfinite(out[k])
        out[k, bad] = out[k - 1, bad]
    empty = ~np.isfinite(out[0])
    if empty.any() and not empty.all():
        _, (jj, ii) = distance_transform_edt(empty, return_indices=True)
        out = out[:, jj, ii]
    return out


def _deep_sponge_columns(Tm: np.ndarray, Sm: np.ndarray, d: np.ndarray, H: np.ndarray, wet: np.ndarray,
                         deep: np.ndarray, tau_days: float, h_min: float, taper_cells: int,
                         period: tuple[str, str], n_rec: int) -> tuple[dict, dict]:
    """Per-column targets of ``deep_sponge(per_column=True)``: (sponge kwargs, summary)."""
    from forcing.ocean_init import source_level_grid
    rate = _deep_sponge_rate(wet, deep, tau_days, taper_cells)
    sp = rate > 0
    zf = source_level_grid(d)["z_f"].values                       # interfaces of all GLORYS levels, surface first
    hmax = float(np.nanmax(np.where(sp, H, np.nan)))
    nk = int(min(d.size, np.searchsorted(-zf[1:], hmax) + 1))    # levels down to the deepest sponge bed
    T3, S3 = _fill_target(Tm[:nk]), _fill_target(Sm[:nk])
    zw = zf[:nk + 1]
    # target water at 40–60 m on the shelf-break columns (500 < H ≤ 1000 m): fraction with S > 35
    zc = 0.5 * (zw[1:] + zw[:-1])
    lev = (zc <= -40.0) & (zc >= -60.0)
    brk = sp & (np.nan_to_num(H) > 500.0) & (np.nan_to_num(H) <= 1000.0)
    sb = S3[lev][:, brk]
    frac = float((sb > 35.0).mean()) if sb.size else float("nan")
    kw = dict(sponge_rate=rate, sponge_T=T3, sponge_zw=zw, sponge_S=S3)
    summary = dict(tau_days=tau_days, h_min=h_min, taper_cells=taper_cells, period=list(period), records=n_rec,
                   per_column=True, sponge_cells=int(deep.sum()), levels=nk,
                   shelfbreak_S35_fraction=frac, shelfbreak_cells=int(brk.sum()),
                   S_surface_range=[float(S3[0, sp].min()), float(S3[0, sp].max())],
                   T_surface_range=[float(T3[0, sp].min()), float(T3[0, sp].max())])
    return kw, summary


def perturb_temperature(T: np.ndarray, H: np.ndarray, amp: float, seed: int = 1) -> np.ndarray:
    """
    Diagnostic only (Gulf Stream drift, figs/gs_drift d11): a copy of the potential temperature T (k, j, i) [°C]
    plus seeded white noise, uniform in [−amp, amp] [°C], in every layer of every wet column (H finite); land
    untouched.  Used to start a twin of a run (``--perturb-T``) and the matching MOM6 initial file (``full_pert``)
    with the same field, to measure intrinsic (chaotic) variability of the free-running path.  Not a parity item.
    """
    if amp < 0:
        raise ValueError(f"amp must be ≥ 0, got {amp}")
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-amp, amp, size=T.shape)
    return np.where(np.isfinite(H)[None], T + noise, T)


def with_paths(case: Case, **kw) -> Case:
    """A copy of ``case`` with some fields replaced (tests)."""
    return replace(case, **kw)
