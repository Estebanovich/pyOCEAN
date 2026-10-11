#!/usr/bin/env python
"""
Parallel-computing benchmark for the PAC_MEX tidal step (650×500).

Measures, on the real grid:
1. the cost of each part of a time step (serial NumPy reference);
2. Numba ``prange`` kernels for the explicit part (Coriolis + drag + AB2 → û,
   and the body tide Σ_c f cos θ Re h − f sin θ Im h) with 1, 2 and 4 threads,
   checked against the NumPy result;
3. the free-surface solve (SuperLU, single-threaded) for reference.

Usage:  NUMBA_NUM_THREADS=4 python bench_parallel.py
Never run with more than 5 threads in total (P-cores only).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numba as nb  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import run_tides as R  # noqa: E402
from core.dynamics.free_surface import ImplicitFreeSurface, open_boundary_hfac  # noqa: E402
from core.grid.horizontal import create_spherical_grid  # noqa: E402
from core.grid.vertical import compute_hfac, create_z_grid  # noqa: E402
from forcing.tides import CONSTITUENTS, body_force, equilibrium_constants, predict  # noqa: E402


def timeit(fn, n=30):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e3          # ms


# ---------------------------------------------------------------- NumPy reference
def explicit_numpy(u, v, f_u, f_v, gx, gy, wet_u, wet_v, hu, hv, gprev_u, gprev_v, dt, cd):
    v_at_u = np.zeros_like(u)
    v_at_u[:, 1:-1] = 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
    u_at_v = np.zeros_like(v)
    u_at_v[1:-1, :] = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
    gu = np.where(wet_u, f_u * v_at_u + gx, 0.0)
    gv = np.where(wet_v, -f_v * u_at_v + gy, 0.0)
    ru = np.where(wet_u, cd * np.hypot(u, v_at_u) / np.where(wet_u, hu, 1.0), 0.0)
    rv = np.where(wet_v, cd * np.hypot(v, u_at_v) / np.where(wet_v, hv, 1.0), 0.0)
    uh = (u + dt * (1.5 * gu - 0.5 * gprev_u)) / (1.0 + dt * ru)
    vh = (v + dt * (1.5 * gv - 0.5 * gprev_v)) / (1.0 + dt * rv)
    return uh, vh, gu, gv


# ---------------------------------------------------------------- Numba kernels
@nb.njit(parallel=True, fastmath=False, cache=True)
def explicit_numba(u, v, f_u, f_v, gx, gy, wet_u, wet_v, hu, hv, gprev_u, gprev_v, dt, cd):
    ny, nxu = u.shape
    nyv, nx = v.shape
    uh = np.zeros_like(u)
    vh = np.zeros_like(v)
    gu = np.zeros_like(u)
    gv = np.zeros_like(v)
    for j in nb.prange(ny):
        for i in range(nxu):
            if wet_u[j, i]:
                vu = 0.0
                if 0 < i < nxu - 1:
                    vu = 0.25 * (v[j, i - 1] + v[j + 1, i - 1] + v[j, i] + v[j + 1, i])
                g = f_u[j, i] * vu + gx[j, i]
                r = cd * np.sqrt(u[j, i] ** 2 + vu ** 2) / hu[j, i]
                gu[j, i] = g
                uh[j, i] = (u[j, i] + dt * (1.5 * g - 0.5 * gprev_u[j, i])) / (1.0 + dt * r)
    for j in nb.prange(nyv):
        for i in range(nx):
            if wet_v[j, i]:
                uv = 0.0
                if 0 < j < nyv - 1:
                    uv = 0.25 * (u[j - 1, i] + u[j - 1, i + 1] + u[j, i] + u[j, i + 1])
                g = -f_v[j, i] * uv + gy[j, i]
                r = cd * np.sqrt(v[j, i] ** 2 + uv ** 2) / hv[j, i]
                gv[j, i] = g
                vh[j, i] = (v[j, i] + dt * (1.5 * g - 0.5 * gprev_v[j, i])) / (1.0 + dt * r)
    return uh, vh, gu, gv


@nb.njit(parallel=True, cache=True)
def body_numba(re, im, fc, fs):
    """η_b = Σ_c (f cos θ)_c Re h_c − (f sin θ)_c Im h_c, re/im (npts, nc)."""
    n, nc = re.shape
    out = np.empty(n)
    for p in nb.prange(n):
        s = 0.0
        for c in range(nc):
            s += fc[c] * re[p, c] - fs[c] * im[p, c]
        out[p] = s
    return out


def main() -> None:
    sd = R.subdomain()
    ny, nx = sd["depth"].shape
    hg = create_spherical_grid(nx, ny, R.DLL, R.DLL, lon0=sd["lon0"], lat0=sd["lat0"])
    vg = create_z_grid(1, [float(sd["depth"].max())])
    hf = open_boundary_hfac(compute_hfac(vg, sd["depth"], hfac_min=0.0), ["west", "east", "south", "north"])
    hu = hf.hfac_u.values[0] * vg.dz.values[0]
    hv = hf.hfac_v.values[0] * vg.dz.values[0]
    wet_u, wet_v = hu > 0, hv > 0
    fl_u = np.zeros_like(wet_u)
    fl_u[:, [0, -1]] = wet_u[:, [0, -1]]
    fl_v = np.zeros_like(wet_v)
    fl_v[[0, -1], :] = wet_v[[0, -1], :]
    fs = ImplicitFreeSurface(hg, vg, hf, dt=120.0, theta=0.6, flather_u=fl_u, flather_v=fl_v)
    f_u = 2 * R.OMEGA_E * np.sin(np.deg2rad(hg["lat_u"].values))
    f_v = 2 * R.OMEGA_E * np.sin(np.deg2rad(hg["lat_v"].values))
    rng = np.random.default_rng(0)
    u = np.where(wet_u, 0.3 * rng.normal(size=wet_u.shape), 0.0)
    v = np.where(wet_v, 0.3 * rng.normal(size=wet_v.shape), 0.0)
    gx, gy = 1e-6 * rng.normal(size=u.shape), 1e-6 * rng.normal(size=v.shape)
    gpu, gpv = 1e-6 * rng.normal(size=u.shape), 1e-6 * rng.normal(size=v.shape)
    hu1, hv1 = np.where(wet_u, hu, 1.0), np.where(wet_v, hv, 1.0)
    cons = list(CONSTITUENTS)
    body = equilibrium_constants(hg.lon_t.values, hg.lat_t.values, cons).reshape(-1, len(cons))
    t = np.datetime64("2026-08-10T03:00")
    eta = np.zeros((ny, nx))

    print(f"grid {nx}×{ny}; NUMBA_NUM_THREADS={nb.config.NUMBA_NUM_THREADS}")
    print("\n1) serial NumPy step parts [ms]")
    parts = {
        "body tide predict (325k×8)": lambda: predict(np.array([t]), body, cons),
        "body_force gradient": lambda: body_force(hg, eta),
        "explicit (Coriolis+drag+AB2)": lambda: explicit_numpy(u, v, f_u, f_v, gx, gy, wet_u, wet_v, hu, hv,
                                                               gpu, gpv, 120.0, R.CD),
        "free-surface step (SuperLU)": lambda: fs.step(eta, u[None], v[None], u[None], v[None],
                                                       flather_eta=(np.zeros_like(u), np.zeros_like(v)),
                                                       flather_ubar=np.zeros_like(u), flather_vbar=np.zeros_like(v)),
    }
    for k, fn in parts.items():
        print(f"   {k:32s} {timeit(fn):7.2f}")

    ref = explicit_numpy(u, v, f_u, f_v, gx, gy, wet_u, wet_v, hu, hv, gpu, gpv, 120.0, R.CD)
    ref_b = predict(np.array([t]), body, cons)[0]
    # basis of the body tide at t (f cos θ, f sin θ per constituent), from predict of unit constants
    fc = predict(np.array([t]), np.eye(len(cons)), cons)[0]
    fsn = -predict(np.array([t]), 1j * np.eye(len(cons)), cons)[0]
    re, im = np.ascontiguousarray(body.real), np.ascontiguousarray(body.imag)

    print("\n2) Numba prange kernels [ms] (max |difference| vs NumPy)")
    for n in (1, 2, 4):
        nb.set_num_threads(n)
        out = explicit_numba(u, v, f_u, f_v, gx, gy, wet_u, wet_v, hu1, hv1, gpu, gpv, 120.0, R.CD)
        err = max(np.max(np.abs(a - b)) for a, b in zip(out, ref))
        te = timeit(lambda: explicit_numba(u, v, f_u, f_v, gx, gy, wet_u, wet_v, hu1, hv1, gpu, gpv, 120.0, R.CD))
        eb = np.max(np.abs(body_numba(re, im, fc, fsn) - ref_b))
        tb = timeit(lambda: body_numba(re, im, fc, fsn))
        print(f"   threads={n}:  explicit {te:6.2f} (err {err:.1e})   body tide {tb:6.2f} (err {eb:.1e})")


if __name__ == "__main__":
    main()
