"""
MITgcm verification/internal_wave (spec 2026-10-12-baroclinic-obc-internal-wave §3): inputs (delXvar,
topog.slope, T.init, data's Tref), the west open-boundary profiles of its code/obcs_calc.F, MITgcm's partial-cell
depth (hFacMin = 0.2) and the MNC reader.  MITgcm imposes T in its boundary cell (column 1, obcs_apply_ts.F:205)
and u at the face east of it (obcs_apply_uv.F:205): the pyOCEAN domain is MITgcm's columns 2 … 60 with its west
face at that u point.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SRC = Path.home() / "pyOCEAN_refs/MITgcm/verification/internal_wave"
RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/internal_wave")
PERIOD = 44567.0                         # obcs_calc.F obTimeScale [s]
U0, N2, ALPHA, G, F0 = 0.024, 1e-6, 2e-4, 9.81, 0.0
NX, NZ, DZ = 60, 20, 10.0


def load_inputs() -> dict:
    """dx (60,), bed depth H (60,) [m] (NaN on land), T (20, 60) and Tref (20,) [°C] surface first, dz (20,)."""
    inp = SRC / "input"
    dx = np.fromfile(inp / "delXvar", ">f8")
    d = np.fromfile(inp / "topog.slope", ">f8")
    H = np.where(d < 0, -d, np.nan)
    T = np.fromfile(inp / "T.init", ">f8").reshape(NZ, NX)
    txt = (inp / "data").read_text()
    tref = np.array([float(v) for v in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?",
                                                  re.search(r"Tref *=([^a-zA-Z]*)", txt).group(1))][:NZ])
    return dict(dx=dx, H=H, T=T, tref=tref, dz=np.full(NZ, DZ))


def mitgcm_depth(H: np.ndarray, dz: np.ndarray, hfac_min: float = 0.2) -> np.ndarray:
    """
    MITgcm's effective depth with partial cells (ini_masks_etc.F:108-120, hFacMinDr = 0): the bottom cell's
    fraction hFac is dropped when < hFacMin/2, else raised to at least hFacMin.
    """
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    Hc = np.asarray(H, float)
    out = np.full(Hc.shape, np.nan)
    ok = np.isfinite(Hc) & (Hc > 0)
    l = np.clip(np.searchsorted(zf, np.where(ok, Hc, 0.0), side="left") - 1, 0, dz.size - 1)
    frac = (Hc - zf[l]) / dz[l]
    frac = np.where(frac < 0.5 * hfac_min, 0.0, np.maximum(frac, hfac_min))
    D = zf[l] + frac * dz[l]
    return np.where(ok & (D > 0), D, np.nan)


def obcs_west(t: float, dz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OBWu (20,) [m/s] and OBWt (20,) [°C], surface first, of internal_wave/code/obcs_calc.F at time t [s]."""
    rF = -np.concatenate([[0.0], np.cumsum(dz)])
    rC = 0.5 * (rF[1:] + rF[:-1])
    vs = np.cos(1.0 * np.pi * (rC / rF[-1]))
    tmpsum = (vs * dz).sum() / rF[-1]                  # obcs_calc.F literally (rF(Nr+1) < 0; Σ vs drF = 0 here)
    vs = vs - tmpsum
    omega = 2.0 * np.pi / PERIOD
    kx = 1.0 * 2.0 * np.pi / 400.0 * np.sqrt((omega**2 - F0**2) / (1e-6 - omega**2))
    dx1 = load_inputs()["dx"][0]
    u = U0 * vs * np.sin(omega * t) * np.cos(kx * (3.0 - 2.0 - 0.5) * dx1)
    strat = 1e-6 / (G * ALPHA)
    k = np.arange(1, dz.size + 1)
    T = load_inputs()["tref"] + U0 * np.sin(np.pi * (k - 0.5) / dz.size) * np.sin(omega * t) \
        * np.sqrt(strat / (ALPHA * G)) * np.sqrt(omega**2 - F0**2) / omega
    return u, T


def read_mitgcm(run: Path | None = None) -> dict:
    """
    MNC state of the MITgcm run: t [s], x centres [m], U (n, 20, 61), T (n, 20, 60), η (n, 60), Depth (60,).
    The case has two tiles along x (sNx = 30, nSx = 2): they are joined by X, the shared u face kept once.
    """
    import glob

    import netCDF4
    run = RUNS / "mitgcm" / "run" if run is None else run
    blocks = sorted({f.rsplit(".t", 1)[0] for f in glob.glob(str(run / "mnc_*" / "state.*.t*.nc"))})
    if not blocks:
        raise FileNotFoundError(f"no MNC state files in {run}")
    t, U, T, eta = [], [], [], []
    for b in blocks:
        tiles = []
        for f in sorted(glob.glob(b + ".t*.nc")):
            with netCDF4.Dataset(f) as ds:
                tiles.append((float(ds["X"][0]), np.asarray(ds["T"][:], float), np.asarray(ds["U"][:, :, 0, :], float),
                              np.asarray(ds["Temp"][:, :, 0, :], float), np.asarray(ds["Eta"][:, 0, :], float),
                              np.asarray(ds["X"][:], float)))
        tiles.sort(key=lambda a: a[0])
        t.append(tiles[0][1])
        U.append(np.concatenate([tiles[0][2]] + [a[2][..., 1:] for a in tiles[1:]], axis=-1))
        T.append(np.concatenate([a[3] for a in tiles], axis=-1))
        eta.append(np.concatenate([a[4] for a in tiles], axis=-1))
        x = np.concatenate([a[5] for a in tiles])
    o = np.argsort(np.concatenate(t))
    cat = lambda a: np.concatenate(a)[o]                                  # noqa: E731
    depth = []
    for f in sorted(glob.glob(str(run / "mnc_*" / "grid.t*.nc"))):
        with netCDF4.Dataset(f) as ds:
            depth.append((float(ds["X"][0]), np.asarray(ds["Depth"][0, :], float)))
    depth = np.concatenate([d for _, d in sorted(depth, key=lambda a: a[0])])
    return dict(t=cat(t), x=x, U=cat(U), T=cat(T), eta=cat(eta), depth=depth)


RHO0 = 999.8                            # MITgcm default rhoNil (not set in data)


def harmonic(t: np.ndarray, x: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit x ≈ a0 + A sin(ωt − φ) along axis 0: amplitude A and phase φ [deg] per point."""
    G_ = np.stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)], 1)
    c = np.linalg.lstsq(G_, x.reshape(t.size, -1), rcond=None)[0]
    amp = np.hypot(c[1], c[2]).reshape(x.shape[1:])
    ph = np.degrees(np.arctan2(-c[2], c[1])).reshape(x.shape[1:])
    return amp, ph


def energy_flux(u, T, eta, H, dz, tref) -> np.ndarray:
    """
    Vertically integrated baroclinic energy flux Σ (u − ū)(p′ − p̄′) Δz [W m⁻¹] at each time and column, from
    level values u, T (n, nz, ncol) surface first, η (n, ncol) and the depth H (ncol,) (levels below the bed
    have NaN): ρ′ = −ρ0 α (T − Tref), p′ = ρ0 g η + g Σ_above ρ′ Δz + g ρ′ Δz/2 (linear EOS, hydrostatic).
    """
    dzz = np.where(np.isfinite(T), dz[None, :, None], 0.0)
    rho = np.nan_to_num(-RHO0 * ALPHA * (T - tref[None, :, None]))
    above = np.cumsum(rho * dzz, axis=1) - rho * dzz
    p = RHO0 * G * eta[:, None, :] + G * (above + 0.5 * rho * dzz)
    D = dzz.sum(axis=1)
    ub = np.nansum(np.nan_to_num(u) * dzz, axis=1) / np.where(D > 0, D, 1.0)
    pb = (p * dzz).sum(axis=1) / np.where(D > 0, D, 1.0)
    return ((np.nan_to_num(u) - ub[:, None]) * (p - pb[:, None]) * dzz).sum(axis=1)


def to_levels(h: np.ndarray, f: np.ndarray, H: np.ndarray, dz: np.ndarray = np.full(NZ, DZ)) -> np.ndarray:
    """pyOCEAN layers (N, …) bed first → the fixed MITgcm levels (20, …) surface first, η in the top level
    (PPM_H4, as validation/bayiw_2d/section.to_mitgcm_levels); NaN below the (partial-cell) bed."""
    from core.vertical import remap as rm
    Hc = np.nan_to_num(np.asarray(H, float))
    zf = np.concatenate([[0.0], np.cumsum(dz)]).reshape((-1,) + (1,) * Hc.ndim)
    top, bot = np.minimum(zf[:-1], Hc), np.minimum(zf[1:], Hc)
    h_lev = (bot - top)[::-1]
    tot, lev = h.sum(axis=0), h_lev.sum(axis=0)
    wet = (tot > 0) & (lev > 0)
    h_tgt = np.where(wet[None], h_lev, 0.0)
    h_tgt[-1] += np.where(wet, tot - lev, 0.0)
    out = rm.remap_columns(np.ascontiguousarray(h, float), np.ascontiguousarray(f, float),
                           np.ascontiguousarray(h_tgt), rm.PPM_H4, wet)
    return np.where(wet[None] & (h_lev > 0), out, np.nan)[::-1]


def boundary_state(dz: np.ndarray):
    """t → {'west': OBData}: the obcs_calc.F profiles on the 20 MITgcm levels at MITgcm's boundary face."""
    from boundaries.obcs import OBData
    zw = -np.concatenate([[0.0], np.cumsum(dz)])[:, None]

    def at(t: float):
        u, T = obcs_west(t, dz)
        return {"west": OBData(side="west", T=T[:, None], u_normal=u[:, None], zw=zw)}
    return at


def build_model(coord: str, dt: float = 500.0):
    """
    pyOCEAN on MITgcm's columns 2 … 60 (west face = MITgcm's prescribed u point), MITgcm's partial-cell
    depth, its viscosities and diffusivities, free slip, f = 0; west 'clamped' with the obcs_calc.F profiles,
    east a wall (MITgcm's eastern open boundary sits on land).  coord: zstar | hybrid | sigma (20 layers).
    """
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    from core.grid.horizontal import create_cartesian_grid_stretched
    from validation.bayiw_baroclinic import bayiw
    d = load_inputs()
    dx = d["dx"][1:]
    H = mitgcm_depth(d["H"][1:], d["dz"])[None, :]
    hg = create_cartesian_grid_stretched(dx, np.array([5e3]), x0=float(d["dx"][0]), y0=0.0, f0=0.0)
    tc = RHO0 * ALPHA
    if coord == "zstar":
        params = dict(dz_top=d["dz"].copy())
    elif coord == "hybrid":
        rho = RHO0 - tc * d["tref"]
        params = dict(dz_top=np.full(NZ, 1.0), rho_target=bayiw.profile_targets(rho, NZ, 0))
    elif coord == "sigma":
        params = {}
    else:
        raise ValueError(f"unknown coord {coord!r}")
    m = SigmaBaroclinic(hg, H, dt=dt, n_layers=NZ, rho0=RHO0, t_coef=tc, t_ref=0.0, z0=None,
                        closure="constant", nu_const=1e-3, ah=1e-2, kh=1e-2, kv=1e-3, pgf="fv", rho_ref="auto",
                        vertical_coordinate="ale", regrid=coord, regrid_params=params,
                        open_boundaries=("west",), open_boundary_state=boundary_state(d["dz"]),
                        open_boundary_mode={"west": "clamped"}, mix_diagnostic=False)
    Hc = np.nan_to_num(H)
    zf = np.concatenate([[0.0], np.cumsum(d["dz"])])[:, None, None]
    h_src = (np.minimum(zf[1:], Hc[None]) - np.minimum(zf[:-1], Hc[None]))[::-1]
    T_src = np.broadcast_to(d["T"][:, None, 1:], h_src.shape)[::-1].copy()
    st = m.initial_state(np.zeros(H.shape), T_src, h_src=h_src)
    return m, st
