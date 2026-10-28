"""
Figures of the manual page theory/pressure-gradient.md.

1. pgf_schematic.png   σ-layers over a sloping bed: the two large terms of the σ-coordinate PGF.
2. pgf_seamount.png    The real PGF functions of SigmaBaroclinic evaluated ONCE on a resting, horizontally
                       stratified Beckmann & Haidvogel (1993) seamount (no time stepping): the exact force is 0,
                       so everything plotted is the error of each scheme.

Run:  NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python docs/manual/figures/make_pgf.py
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

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, reference_profile  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402

OUT = ROOT / "docs/manual/img/pressure-gradient"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "grey": "#666666"}


# ----------------------------------------------------------------------------- 1. schematic
def schematic() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.linspace(0.0, 10.0, 200)
    H = 600.0 - 40.0 * x                          # bed depth [m]: 600 m → 200 m
    N = 6
    for k in range(N + 1):
        s = k / N
        ax.plot(x, -H + s * H, color=C["grey"], lw=0.8 if 0 < k < N else 1.4)
    ax.fill_between(x, -H, -650, color="#d9c9a3")
    for zi in (-100.0, -250.0, -400.0):           # horizontal isopycnals
        ax.plot(x, np.full_like(x, zi), color=C["blue"], lw=1.2, ls="--")
    ax.text(9.9, -95, "isopycnals (ρ = ρ(z))", color=C["blue"], ha="right", va="bottom")
    # two columns and a σ-surface
    xL, xR = 4.0, 6.0
    k = 2
    zL = -(600 - 40 * xL) * (1 - k / N)
    zR = -(600 - 40 * xR) * (1 - k / N)
    for xc in (xL, xR):
        ax.axvline(xc, color="k", lw=0.6, ls=":")
    ax.plot([xL, xR], [zL, zR], "o-", color=C["red"], lw=2.0, ms=6)
    ax.annotate("", xy=(xR, zR), xytext=(xR, zL), arrowprops=dict(arrowstyle="<->", color=C["red"]))
    ax.text(xR + 0.1, 0.5 * (zL + zR), r"$\delta_x z|_\sigma$", color=C["red"], va="center")
    ax.text(4.6, zL + 30, r"σ-surface $k$", color=C["red"], ha="center")
    ax.text(0.2, -600,
            r"$-\frac{1}{\rho_0}\left.\frac{\partial P}{\partial x}\right|_z = "
            r"-\frac{1}{\rho_0}\left[\left.\frac{\partial P}{\partial x}\right|_\sigma"
            r"\;+\;g\rho\left.\frac{\partial z}{\partial x}\right|_\sigma\right]$"
            "\n" r"two large terms of opposite sign; their sum is the small physical force",
            fontsize=10, va="bottom", bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.95))
    ax.set_xlim(0, 10)
    ax.set_ylim(-650, 20)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("z [m]")
    ax.set_title("σ-layers over a sloping bed: at rest the two terms must cancel exactly")
    fig.tight_layout()
    fig.savefig(OUT / "pgf_schematic.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- 2. seamount at rest
RHO0, ALPHA = 1000.0, 0.2


def seamount(pgf: str, nx=48, ny=40, nz=20, dx=6000.0, coord="sigma"):
    """Beckmann & Haidvogel (1993) seamount of validation/seamount (H0 4500 m, 4050 m high, L = 25 km)."""
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=1e-4)
    x = (np.arange(nx) + 0.5) * dx - nx * dx / 2
    y = (np.arange(ny) + 0.5) * dx - ny * dx / 2
    H = 4500.0 - 4050.0 * np.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / 25e3**2)
    kw = {}
    if coord == "zstar":
        kw = dict(vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(nz, 4500.0 / nz)))
    m = SigmaBaroclinic(hg, H, dt=600.0, n_layers=nz, rho0=RHO0, t_coef=ALPHA, z0=None, closure="constant",
                        nu_const=1e-4, pgf=pgf, **kw)
    return m, x, H


def resting_T(z):
    rho = RHO0 + 3.0 * (1.0 - np.exp(z / 1000.0))            # validation/seamount/README.md
    return (RHO0 - rho) / ALPHA, rho


def evaluate():
    cases = [("twoterm", False, "sigma", "two-term, ρ₀"),
             ("jacobian", False, "sigma", "Jacobian, ρ₀"),
             ("fv", False, "sigma", "FV, ρ₀"),
             ("jacobian", True, "sigma", "Jacobian + ρ_ref"),
             ("fv", True, "sigma", "FV + ρ_ref"),
             ("fv", False, "zstar", "z* + FV, ρ₀"),
             ("fv", True, "zstar", "z* + FV + ρ_ref")]
    res = {}
    for pgf, ref, coord, label in cases:
        m, x, H = seamount(pgf, coord=coord)
        eta = np.zeros(H.shape)
        if coord == "zstar":
            st = m.initial_state(eta, np.zeros((m.n,) + H.shape))     # builds h (z* grid)
            h = st.h
            z = m.layer_heights(eta, h)
        else:
            h = None
            z = m.layer_heights(eta)
        T, rho = resting_T(z)
        if ref:
            act = np.broadcast_to(m.fs.active[None], z.shape)
            m.rho_ref = reference_profile(z[act], rho[act])        # as rho_ref="auto" does in initial_state
        gx, gy = m._baroclinic_pgf(eta, T, h)
        if coord == "zstar":                                       # report only faces that carry water (≥ 1 cm)
            hu, hv = m._mean_face_thickness(h)
            gx = np.where(hu >= 0.01, gx, 0.0)
            gy = np.where(hv >= 0.01, gy, 0.0)
        res[label] = dict(gx=gx, err=max(np.abs(gx).max(), np.abs(gy).max()), x=x, H=H, z=z, m=m)
        print(f"{label:22s} max|PGF| = {res[label]['err']:.3e} m s^-2")
    return res


def plot_seamount(res) -> None:
    fig = plt.figure(figsize=(10.0, 4.4))
    ax = fig.add_subplot(1, 2, 1)
    labels = list(res)
    vals = [res[k]["err"] for k in labels]
    cols = [C["grey"], C["orange"], C["sky"], C["red"], C["blue"], C["purple"], C["green"]]
    ax.barh(range(len(labels)), vals, color=cols)
    ax.set_xscale("log")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"max $|$PGF error$|$ at rest [m s$^{-2}$]")
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, f"{v:.1e}", va="center", fontsize=9)
    ax.set_xlim(min(vals) / 3, max(vals) * 30)
    ax.set_title("(a) one evaluation, seamount at rest\n(z*: faces ≥ 1 cm thick)")
    # section through the summit row: Jacobian without and with ρ_ref
    ax2 = fig.add_subplot(1, 2, 2)
    for label, cmap_lim, col in (("Jacobian, ρ₀", None, C["orange"]), ("Jacobian + ρ_ref", None, C["red"])):
        r = res[label]
        j = r["gx"].shape[1] // 2
        g = np.abs(r["gx"][:, j, :])                                 # (N, nx+1)
        xf = np.concatenate([[r["x"][0] - 3e3], 0.5 * (r["x"][1:] + r["x"][:-1]), [r["x"][-1] + 3e3]]) / 1e3
        ax2.semilogy(xf, g.max(axis=0) + 1e-16, color=col, lw=1.6, label=label)
    r = res["z* + FV + ρ_ref"]
    j = r["gx"].shape[1] // 2
    ax2.semilogy(xf, np.abs(r["gx"][:, j, :]).max(axis=0) + 1e-16, color=C["green"], lw=1.6, label="z* + FV + ρ_ref")
    ax2.set_xlabel("x through the summit [km]")
    ax2.set_ylabel(r"max over layers of $|$PGF$_x|$ [m s$^{-2}$]")
    ax2.set_ylim(1e-10, 1e-3)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_title("(b) error along the section y = 0")
    fig.tight_layout()
    fig.savefig(OUT / "pgf_seamount.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    schematic()
    plot_seamount(evaluate())
    print("written:", sorted(p.name for p in OUT.glob("*.png")))
