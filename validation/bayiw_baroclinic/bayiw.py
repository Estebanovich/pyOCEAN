"""
Shared setup of the BayIW baroclinic benchmark against MITgcm (spec
docs/superpowers/specs/2026-10-12-bayiw-baroclinic-benchmark-design.md): the MITgcm inputs of
Esteban's M.Sc. runs (BayIW_Rectan_2layer, BayIW_Rectan_linear; run_expand/data, data.obcs), the
ALE model built with the same parameters, and depth sampling on moving layers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid_stretched

NX, NY = 560, 352
F0, RHO0, ALPHA = 6.97e-5, 999.8, 2e-4                  # MITgcm data: f0, rhoNil, tAlpha
T_COEF = ALPHA * RHO0                                   # ρ = ρ₀ − ρ₀α (T − tRef) [kg m⁻³ °C⁻¹]
DELR = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5, 3.8, 4.1,
                 4.4, 4.8, 5.2, 5.6, 6.0, 6.5, 7.1, 7.6, 8.2, 8.9, 9.6, 10.4, 11.3, 12.2, 13.2, 14.3, 15.4, 16.7,
                 18.0, 19.5, 21.1, 22.8, 24.6, 26.6, 28.8, 31.1, 33.7, 36.4, 39.3, 42.5, 45.8])   # PARM04, surface first [m]
Z_F = -np.concatenate([[0.0], np.cumsum(DELR)])         # MITgcm interface heights, surface first [m]
Z_C = 0.5 * (Z_F[:-1] + Z_F[1:])                        # level centres RC [m]
Z110, Z250 = Z_C[29], Z_C[39]                           # z_ref of the 110 m and 250 m reference series
X0, Y0 = -1400e3, -1621e3                               # xgOrigin, ygOrigin [m]
STATIONS = {"Seiche": (280, 345), "Seiche2": (280, 325), "A": (230, 315), "B": (330, 315),
            "C": (280, 265), "Gamma": (280, 215), "Omega": (280, 315)}           # (i, j), Output_IWBay meta
ZOOM_I0, ZOOM_NI = 130, 300                             # zoom arrays: columns i = 130 … 429, all rows
GAMMA_I, GAMMA_J0 = 280, 215                            # YZ section along i = 280 from Gamma (j = 215 … 351)
REF = Path("/Volumes/ESTEBAN_MIT/MITgcm/verification/Output_IWBay/npz_out")
RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs")
CASES = {
    "2layer": dict(dir=Path("/Volumes/ESTEBAN_MIT/MITgcm/verification/BayIW_Rectan_2layer"),
                   temp="T_2layer_50z_560x352.bin", t_ref=11.4),
    "linear": dict(dir=Path("/Volumes/Esteban_SSD/Maestría/MITgcm/verification/BayIW_Rectan_linear"),
                   temp="linear_temp_50zlev_560x352.bin", t_ref=33.6),
}
STEMS = {"bay": "bahia_01_expand", "nobay": "nobahia_01_expand"}
N_LIGHT = 10                                            # hybrid: top z* layers (Σ delR[:10] = 14.7 m)


def read_bin(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    """MITgcm binary input: big-endian float64 (readBinaryPrec = 64)."""
    return np.fromfile(path, ">f8").reshape(shape)


def cell_centres(d: np.ndarray, origin: float) -> np.ndarray:
    """Cell-centre coordinates [m] of a stretched axis from its spacings (MITgcm XC/YC)."""
    return origin + np.cumsum(d) - 0.5 * d


def mitgcm_depth(H: np.ndarray) -> np.ndarray:
    """
    MITgcm's effective depth [m] of bed depths H [m] (NaN = land) on the delR levels: clipped at
    Σ delR = 599.9 m and, with hFacMin = 1 (set_defaults.F:180; not set in Esteban's data), the bottom cell
    dropped when its fraction hFac < ½ and made full otherwise (ini_masks_etc.F:108-120), i.e. rounded to
    the nearest level interface.  Depth 0 → land (NaN).
    """
    d = -Z_F                                                             # interface depths, increasing
    Hc = np.minimum(np.asarray(H, float), d[-1])
    ok = np.isfinite(Hc) & (Hc > 0)
    l = np.clip(np.searchsorted(d, np.where(ok, Hc, 0.0), side="left") - 1, 0, DELR.size - 1)
    frac = (Hc - d[l]) / DELR[l]
    D = np.where(frac < 0.5, d[l], d[l + 1])
    return np.where(ok & (D > 0), D, np.nan)


def load_inputs(case: str, bay: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MITgcm inputs: dx (NX,), dy (NY,) [m]; rest depth H (NY, NX) [m], NaN on land: MITgcm's effective depth,
    clipped at Σ delR = 599.9 m (intentional, confirmed by Esteban) and rounded to a level interface
    (mitgcm_depth); initial T (50, NY, NX) on the delR levels [°C];
    meridional wind stress records (nrec, NY, NX) at the v-faces [N m⁻²].
    """
    inp = CASES[case]["dir"] / "input"
    stem = STEMS[bay]
    dx = np.fromfile(inp / f"{stem}_dx.bin", ">f8")
    dy = np.fromfile(inp / f"{stem}_dy.bin", ">f8")
    B = read_bin(inp / f"{stem}_bat.bin", (NY, NX))
    H = mitgcm_depth(np.where(B < 0, -B, np.nan))
    T = read_bin(inp / CASES[case]["temp"], (50, NY, NX))
    wind = read_bin(inp / "merid_local_bay_560x352.bin", (-1, NY, NX))
    return dx, dy, H, T, wind


def mitgcm_profile_at(z: np.ndarray, T_mit: np.ndarray) -> np.ndarray:
    """T of the MITgcm level containing each height z (N, NY, NX) [m]: piecewise constant, so a two-layer step stays sharp."""
    k = np.clip(np.searchsorted(-Z_F, -z, side="right") - 1, 0, DELR.size - 1)
    return np.take_along_axis(T_mit, k, axis=0)


def mitgcm_source(H: np.ndarray, T_mit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The MITgcm z-levels as the source grid of the ALE initial state (SigmaBaroclinic.initial_state h_src):
    thicknesses (50, …) [m] of delR clipped at the bed H (NaN = land → 0), bed first, and T_mit flipped to
    bed first.  Every model layer then gets the exact level means, also where hybrid layers do not line
    up with the levels.
    """
    Hc = np.nan_to_num(np.asarray(H, float))[None]
    zf = Z_F.reshape((-1,) + (1,) * (Hc.ndim - 1))                         # interfaces, surface first
    top, bot = np.minimum(-zf[:-1], Hc), np.minimum(-zf[1:], Hc)          # depths of each level's top and bottom
    return (bot - top)[::-1].copy(), np.asarray(T_mit, float)[::-1].copy()


def sample_at_depth(zc: np.ndarray, f: np.ndarray, h: np.ndarray, z0: float, bed: np.ndarray) -> np.ndarray:
    """
    f at height z0 [m] by linear interpolation in z between layer centres zc (N, …), bed first, skipping
    layers thinner than 1e-6 m (vanished z* layers): constant between the bed and the deepest centre and
    above the top centre; NaN below the bed ``bed`` (…) = −H and in columns without water.
    """
    n = zc.shape[0]
    ok = h > 1e-6
    k = np.arange(n).reshape((n,) + (1,) * (zc.ndim - 1))
    kb = np.where(ok & (zc <= z0), k, -1).max(axis=0)
    ka = np.where(ok & (zc > z0), k, n).min(axis=0)
    has_b, has_a = kb >= 0, ka < n
    take = lambda a, i: np.take_along_axis(a, np.clip(i, 0, n - 1)[None], axis=0)[0]   # noqa: E731
    zb, za, fb, fa = take(zc, kb), take(zc, ka), take(f, kb), take(f, ka)
    both = has_b & has_a
    w = np.where(both, (z0 - zb) / np.where(both & (za > zb), za - zb, 1.0), 0.0)
    out = np.where(both, fb + w * (fa - fb), np.where(has_b, fb, fa))
    return np.where(ok.any(axis=0) & (z0 >= bed), out, np.nan)


def centred_targets(rho_a: float, rho_b: float, n: int, n_light: int = 0) -> np.ndarray:
    """
    n + 1 interface target densities with two water masses ρ_a < ρ_b at the centres of brackets n_light + 1
    and n − 2 (a density exactly on a target can split a water mass between two layers, as in the 21-layer
    lock exchange); the first n_light brackets are lighter than any water, so the hybrid rule keeps those
    layers on their z* floor (OM4 style, Adcroft et al. 2019).
    """
    s = (rho_b - rho_a) / (n - n_light - 3)
    return rho_a - (1.5 + n_light) * s + s * np.arange(n + 1)


def profile_targets(rho: np.ndarray, n: int, n_light: int = 0) -> np.ndarray:
    """
    n + 1 targets: n − n_light + 1 evenly spaced over the density range, half an interval beyond each
    extreme, preceded by n_light lighter ones at the same spacing (z* layers at the top, as centred_targets).
    """
    lo, hi = float(np.nanmin(rho)), float(np.nanmax(rho))
    m = n - n_light
    s = (hi - lo) / m
    core = np.linspace(lo - 0.5 * s, hi + 0.5 * s, m + 1)
    return np.concatenate([core[0] - s * np.arange(n_light, 0, -1), core])


def build_model(case: str, bay: str, coord: str = "zstar", dt: float = 120.0):
    """
    The ALE model with the MITgcm parameters (spec §3): z* on the 50 delR levels, or hybrid with 25 layers:
    floor dz_top = the first 25 delR and N_LIGHT = 10 targets lighter than any water, so the top 10 layers
    (14.7 m) stay z* and the other 15 are isopycnal (the hybrid rule makes z layers only where the target is
    lighter than the water).  Returns (model, initial state, wind records).
    """
    dx, dy, H, T_mit, wind = load_inputs(case, bay)
    hg = create_cartesian_grid_stretched(dx, dy, x0=X0, y0=Y0, f0=F0)
    t_ref = CASES[case]["t_ref"]
    if coord == "zstar":
        nz, params = DELR.size, dict(dz_top=DELR.copy())
    elif coord == "hybrid":
        nz = 25
        rho = RHO0 - T_COEF * (T_mit[:, np.isfinite(H)] - t_ref)
        tgt = (centred_targets(float(rho.min()), float(rho.max()), nz, N_LIGHT) if case == "2layer"
               else profile_targets(rho, nz, N_LIGHT))
        params = dict(dz_top=DELR[:nz].copy(), rho_target=tgt)
    else:
        raise ValueError(f"unknown coord {coord!r}")
    m = SigmaBaroclinic(hg, H, dt=dt, n_layers=nz, rho0=RHO0, t_coef=T_COEF, t_ref=t_ref, z0=None,
                        closure="constant", nu_const=1e-5, ah=100.0, kh=100.0, kv=1e-5, pgf="fv", rho_ref="auto",
                        vertical_coordinate="ale", regrid=coord, regrid_params=params,
                        open_boundaries=("south", "west", "east"), orlanski_cmax=0.45, sponge_cells=30,
                        sponge_tau=(1200.0, 60.0), sponge_tracer=False)
    h_src, T_src = mitgcm_source(H, T_mit)                               # remap from the MITgcm levels
    st = m.initial_state(np.zeros(H.shape), T_src, h_src=h_src)
    return m, st, wind
