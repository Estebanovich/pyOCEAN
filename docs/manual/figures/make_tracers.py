"""
Figure of the manual page theory/tracers.md: 1-D advection of a square wave and a Gaussian over one
period of a periodic domain, with the REAL face-reconstruction functions of the stratified model
(core.dynamics.sigma_baroclinic: _up3_faces for 'upwind1'/'upwind3', _weno5_faces_numpy for 'weno5js'/'weno5')
and the model's SSP-RK3 stage formulas (_advect_numpy) on a uniform velocity; WENO5-Z is also shown with the
Zalesak (1979) FCT of the ALE step, built exactly as _advect_fct_numpy (RK3 high-order flux, donor-cell
low-order flux) with the model's coefficient function fct_coefficients_numpy.

Periodicity: each call pads the row with 3 ghost cells on each side (all valid), so the code's
wall rules (lower order next to walls) never act; the faces of the real domain are cut out.

Run:  NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python docs/manual/figures/make_tracers.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.dynamics.sigma_baroclinic import _up3_faces, _weno5_faces_numpy, fct_coefficients_numpy  # noqa: E402

OUT = ROOT / "docs/manual/img/tracers"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})

NX, U, DX, CFL = 200, 1.0, 1.0, 0.4
DT = CFL * DX / U
NSTEP = int(round(NX * DX / U / DT))            # one period
P = 3                                           # ghost cells


def initial() -> np.ndarray:
    x = np.arange(NX) + 0.5
    sq = np.where((x > 20) & (x < 60), 1.0, 0.0)
    ga = np.exp(-((x - 130.0) / 10.0) ** 2)
    return sq + ga


def faces(c: np.ndarray, scheme: str) -> np.ndarray:
    """Face values on the NX+1 faces of the periodic row (face 0 = face NX), u > 0."""
    cp = np.concatenate([c[-P:], c, c[:P]])
    flux = np.full(cp.size + 1, U)
    ok = np.ones(cp.size, bool)
    if scheme in ("upwind1", "upwind3"):
        f = _up3_faces(cp, flux, ok, axis=0, order=1 if scheme == "upwind1" else 3)
    else:
        f = _weno5_faces_numpy(cp, flux, ok, axis=0, variant="js" if scheme == "weno5js" else "z")
    return f[P:P + NX + 1]


def flux(c, scheme):
    return U * faces(c, scheme)                 # F = Q c_face with Q = u Δy Δz = u (unit cell)


def div(F):
    return F[1:] - F[:-1]


def rk3_step(c, scheme):
    """SSP-RK3 of SigmaBaroclinic._advect_numpy with V0 = V1 = 1 (unit cells)."""
    q1 = c - DT * div(flux(c, scheme))
    q2 = 0.75 * c + 0.25 * (q1 - DT * div(flux(q1, scheme)))
    return c / 3.0 + 2.0 / 3.0 * (q2 - DT * div(flux(q2, scheme)))


def fct_step(c, scheme="weno5"):
    """_advect_fct_numpy in 1-D: F_H = (F0 + F1)/6 + 2F2/3, F_L donor cell, Zalesak limiter on A = F_H − F_L."""
    f0 = flux(c, scheme)
    c1 = c - DT * div(f0)
    f1 = flux(c1, scheme)
    c2 = 0.75 * c + 0.25 * (c1 - DT * div(f1))
    f2 = flux(c2, scheme)
    FH = (f0 + f1) / 6.0 + 2.0 * f2 / 3.0
    FL = U * faces(c, "upwind1")
    A = FH - FL
    cL = c - DT * div(FL)
    # the model's coefficient function on a periodically padded stack (nt=1, N=1, ny=1, nx)
    pad = lambda a: np.concatenate([a[-P:], a, a[:P]])[None, None, None]           # noqa: E731
    idx = (np.arange(NX + 2 * P + 1) - P) % NX
    Ax = A[idx][None, None, None]
    Ay = np.zeros((1, 1, 2, NX + 2 * P))
    V1 = np.ones((1, 1, NX + 2 * P))
    Cx, _ = fct_coefficients_numpy(pad(c), pad(cL), V1, Ax, Ay, np.ones((1, 1, NX + 2 * P), bool), DT)
    Cf = Cx[0, 0, 0, P:P + NX + 1]
    return c - DT * div(FL + Cf * A)


def run(step, scheme):
    c = initial()
    for _ in range(NSTEP):
        c = step(c, scheme)
    return c


def main() -> None:
    c0 = initial()
    x = np.arange(NX) + 0.5
    runs = [("upwind1", rk3_step, "upwind1", "#666666"),
            ("upwind3", rk3_step, "upwind3", "#E69F00"),
            ("weno5js (WENO5-JS)", rk3_step, "weno5js", "#56B4E9"),
            ("weno5 (WENO5-Z)", rk3_step, "weno5", "#0072B2"),
            ("weno5 + FCT", fct_step, "weno5", "#D55E00")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]
    ax.plot(x, c0, color="k", lw=1.0, ls="--", label="exact")
    lines = []
    for label, step, scheme, col in runs:
        c = run(step, scheme)
        l1 = np.abs(c - c0).sum() / np.abs(c0).sum()
        lines.append((label, c.min(), c.max(), l1))
        ax.plot(x, c, color=col, lw=1.4, label=f"{label}: L1 {l1:.3f}")
        print(f"{label:20s} min {c.min():+.2e} max {c.max():.4f}  relative L1 {l1:.4f}  sum {c.sum():.10f}")
    ax.set_xlabel("x [cells]")
    ax.set_ylabel("tracer")
    ax.set_title(f"one period, Courant {CFL}, SSP-RK3 ({NSTEP} steps)")
    ax.legend(fontsize=8.5, loc="center", bbox_to_anchor=(0.47, 0.42))
    ax.set_ylim(-0.15, 1.25)
    ax2 = axes[1]
    for label, step, scheme, col in runs:
        c = run(step, scheme)
        ax2.plot(x, c, color=col, lw=1.4)
    ax2.plot(x, c0, color="k", lw=1.0, ls="--")
    ax2.set_xlim(52, 66)
    ax2.set_ylim(-0.06, 1.08)
    ax2.set_xlabel("x [cells]")
    ax2.set_title("zoom: downstream edge of the square")
    fig.tight_layout()
    fig.savefig(OUT / "advection_1d.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
