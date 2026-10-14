"""
Case definitions of the four-model benchmark, part A (spec 2026-10-13-ilicak-four-model-benchmark-design §A1).
One definition per case; every model's writer and pyOCEAN's runner read it.

- lock exchange: Ilicak et al. (2012, §3.1, eq. 3–4) with T = 5/30 °C as Petersen et al. (2015, eq. 21–22) and the
  MOM6 lock_exchange example (Ilicak's 35 °C is inconsistent with Δρ = 5 at α = 0.2);
- overflow: Ilicak (2012, §4) with the shelf of Petersen (2015, eq. 24–25, Table 5);
- internal wave: Petersen (2015, eq. 26–27), high amplitude A = 2 °C (Ilicak Fig. 11b); N = 6.34e-3 s⁻¹ here,
  Ilicak quotes 6.26e-3;
- seamount: Beckmann & Haidvogel (1993, §2c), the definition of validation/seamount.

Linear EOS ρ = ρ₀ − α (T − T_ref) [kg m⁻³], ρ₀ = 1000, α = 0.2 kg m⁻³ °C⁻¹, T_ref = 0 °C.
Coordinates: x, y [m] from the west/south edge of the domain; z [m] ≤ 0, up from the rest surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from validation.seamount import run_seamount as _sm

RHO0, ALPHA, T_REF, G, NU_V = 1000.0, 0.2, 0.0, 9.81, 1e-4
RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark")


def density(T: np.ndarray) -> np.ndarray:
    """Linear EOS ρ = ρ₀ − α (T − T_ref) [kg m⁻³] (Vallis 2006, eq. 1.156 without S)."""
    return RHO0 - ALPHA * (np.asarray(T, float) - T_REF)


@dataclass(frozen=True)
class Case:
    name: str
    dx: np.ndarray                 # (nx,) [m]
    dy: np.ndarray                 # (ny,) [m]
    H: np.ndarray                  # (ny, nx) rest depth [m], positive
    dz: np.ndarray                 # (nz,) nominal z-level thicknesses [m], surface first
    T_fn: Callable                 # T(x, y, z) [°C], broadcasting
    nu_h: tuple                    # horizontal viscosities of the sweep [m² s⁻¹]
    dt: float                      # results time step [s]
    hours: float                   # run length [h]
    out_s: float                   # output interval [s]
    f0: float = 0.0                # Coriolis [s⁻¹]

    @property
    def nx(self) -> int:
        return self.dx.size

    @property
    def ny(self) -> int:
        return self.dy.size

    @property
    def nz(self) -> int:
        return self.dz.size

    @property
    def x(self) -> np.ndarray:
        return np.cumsum(self.dx) - 0.5 * self.dx

    @property
    def y(self) -> np.ndarray:
        return np.cumsum(self.dy) - 0.5 * self.dy

    @property
    def zc(self) -> np.ndarray:
        zf = -np.concatenate([[0.0], np.cumsum(self.dz)])
        return 0.5 * (zf[1:] + zf[:-1])


def _lock() -> Case:
    nx = 128
    T = lambda x, y, z: np.broadcast_to(np.where(np.asarray(x) < 32e3, 5.0, 30.0),            # noqa: E731
                                        np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))).copy()
    return Case("lock", np.full(nx, 500.0), np.array([500.0]), np.full((1, nx), 20.0), np.full(20, 1.0), T,
                (0.1, 10.0, 100.0), 10.0, 17.0, 900.0)


def _overflow() -> Case:
    nx, d1, d2, x0, r = 200, 500.0, 2000.0, 40e3, 7e3
    x = np.arange(nx) * 1e3 + 500.0
    H = d1 + 0.5 * (d2 - d1) * (1.0 + np.tanh((x - x0) / r))                                 # Petersen eq. 24
    T = lambda x, y, z: np.broadcast_to(np.where(np.asarray(x) < 20e3, 10.0, 20.0),           # noqa: E731
                                        np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))).copy()
    return Case("overflow", np.full(nx, 1e3), np.array([1e3]), H[None, :], np.full(100, 20.0), T,
                (1.0, 100.0, 1000.0), 10.0, 40.0, 3600.0)


def _iw() -> Case:
    t_bot, t_top, z_bot, A, L, x0, dz = 10.1, 20.1, -487.5, 2.0, 50e3, 125e3, 25.0

    def T(x, y, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        base = t_bot + (t_top - t_bot) * (z_bot - z) / z_bot                                     # Petersen eq. 26
        bump = np.where(np.abs(x - x0) < L, -A * np.cos(np.pi * (x - x0) / (2 * L))
                        * np.sin(np.pi * (z + dz / 2) / (z_bot + dz / 2)), 0.0)                  # eq. 27
        return np.broadcast_to(base + bump, np.broadcast_shapes(x.shape, np.shape(y), z.shape)).copy()

    return Case("iw", np.full(50, 5e3), np.array([5e3]), np.full((1, 50), 500.0), np.full(20, dz), T,
                (0.01, 15.0, 150.0), 300.0, 200 * 24.0, 6 * 3600.0)


def _seamount() -> Case:
    dx, nx, ny = 6e3, 64, 48
    xc = (np.arange(nx) + 0.5) * dx - nx * dx / 2
    yc = (np.arange(ny) + 0.5) * dx - ny * dx / 2
    H = _sm.H0 - _sm.HS * np.exp(-(xc[None, :] ** 2 + yc[:, None] ** 2) / _sm.LS ** 2)
    T = lambda x, y, z: np.broadcast_to(                                                      # noqa: E731
        (RHO0 - (_sm.RHO0 + _sm.DRHO * (1 - np.exp(np.asarray(z, float) / _sm.DECAY)))) / ALPHA + T_REF,
        np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))).copy()
    return Case("seamount", np.full(nx, dx), np.full(ny, dx), H, np.full(20, _sm.H0 / 20), T,
                (0.0,), 600.0, 240.0, 86400.0, f0=_sm.F0)


_BUILDERS = {"lock": _lock, "overflow": _overflow, "iw": _iw, "seamount": _seamount}


def get(name: str) -> Case:
    """The case ``name`` ∈ {'lock', 'overflow', 'iw', 'seamount'}."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown case {name!r}; expected one of {sorted(_BUILDERS)}")
    return _BUILDERS[name]()


def targets(case: Case, n: int) -> np.ndarray:
    """n+1 interface target densities [kg m⁻³], surface first, increasing, evenly spaced, with the lightest and the
    densest initial water at the centres of the first and last layers (MOM6 COORD_CONFIG 'layer_ref':
    LIGHTEST_DENSITY = ρ_min, DENSITY_RANGE = ρ_max − ρ_min, MOM_coord_initialization.F90:164-200)."""
    T = case.T_fn(case.x[None, None, :], case.y[None, :, None], case.zc[:, None, None])
    rho = density(T)
    lo, hi = float(rho.min()), float(rho.max())
    d = (hi - lo) / (n - 1)
    return np.linspace(lo - 0.5 * d, hi + 0.5 * d, n + 1)


def run_dir(case: str, model: str, coord: str, nu: float, tag: str = "") -> Path:
    """RUNS/<case>/<model>_<coord>_nu<nu>[_<tag>]/ — one run, holding series.npz."""
    return RUNS / case / (f"{model}_{coord}_nu{nu:g}" + (f"_{tag}" if tag else ""))
