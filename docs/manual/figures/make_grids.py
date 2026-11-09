"""
Figures of the manual page theory/grids.md.

Writes PNGs to docs/manual/img/grids/:
  cgrid_stencil.png      Arakawa C-grid cell with the variable locations and metric names
  sigma_stretching.png   Song & Haidvogel (1994) stretched σ-levels for several θ_s, θ_b (core.grid.sigma_levels)
  zlevel_hfac.png        z-levels with partial cells (core.grid.vertical.compute_hfac) over a slope
  slope_factors.png      rx0 and rx1 on a shelf-break section (diagnostics.domain_check.rx0 / rx1)
  bathy_pipeline.png     smoothing and isolated-cell removal (core.grid.bathymetry)

Cheap function calls only (no time stepping).  Run from anywhere:
    python docs/manual/figures/make_grids.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from core.grid.bathymetry import fill_isolated_cells, smooth_bathymetry  # noqa: E402
from core.grid.sigma_levels import sigma_from_thicknesses, stretched_sigma  # noqa: E402
from core.grid.vertical import compute_hfac, create_z_grid  # noqa: E402
from diagnostics.domain_check import rx0, rx1  # noqa: E402

OUT = ROOT / "docs" / "manual" / "img" / "grids"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 9})

# Okabe–Ito colour-blind-safe palette
C_T, C_U, C_V, C_Q, C_GREY = "#000000", "#0072B2", "#D55E00", "#009E73", "#999999"
C_Y, C_P, C_S = "#E69F00", "#CC79A7", "#56B4E9"


def cgrid_stencil() -> None:
    """One T-cell and its neighbours, with T, u, v, q points and the metric names of core/grid/horizontal.py."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for x in (0, 1, 2):
        ax.plot([x, x], [-0.15, 2.15], color=C_GREY, lw=0.8)
    for y in (0, 1, 2):
        ax.plot([-0.15, 2.15], [y, y], color=C_GREY, lw=0.8)
    ax.add_patch(Rectangle((1, 1), 1, 1, facecolor="#E8E8F4", edgecolor="k", lw=1.6, zorder=0))
    for i in range(2):
        for j in range(2):
            ax.plot(i + 0.5, j + 0.5, "o", color=C_T, ms=9)
            ax.text(i + 0.56, j + 0.56, f"T[{j},{i}]", fontsize=9)
    for i in range(3):
        for j in range(2):
            ax.plot(i, j + 0.5, ">", color=C_U, ms=10)
            ax.text(i + 0.04, j + 0.40, f"u[{j},{i}]", color=C_U, fontsize=9)
    for i in range(2):
        for j in range(3):
            ax.plot(i + 0.5, j, "^", color=C_V, ms=10)
            ax.text(i + 0.55, j + 0.05, f"v[{j},{i}]", color=C_V, fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.plot(i, j, "s", color=C_Q, ms=7)
    ax.text(2.04, 2.04, "q[2,2]: f_q, ζ", color=C_Q, fontsize=9)

    def dim(x0, y0, x1, y1, text, color, off=(0, 0), **kw):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<->", mutation_scale=10, color=color, lw=1.3))
        ax.text(0.5 * (x0 + x1) + off[0], 0.5 * (y0 + y1) + off[1], text, color=color, ha="center", va="center",
                fontsize=9, **kw)

    # T-cell (1,1): dx_t, dy_t, area_t
    dim(1, 1.85, 2, 1.85, "dx_t[1,1]", C_T, (0, 0.08))
    dim(1.85, 1, 1.85, 2, "dy_t[1,1]", C_T, (-0.1, 0.25), rotation=90)
    ax.text(1.5, 1.25, "area_t = dx_t·dy_t\nf_t, mask_t, H", ha="center", fontsize=9)
    # u-face (j=0, i=1): dx_u = distance between T-points, dy_u = face length
    dim(0.5, 0.3, 1.5, 0.3, "dx_u[0,1]  (T→T distance, dxC)", C_U, (0, -0.08))
    dim(1.0, 0.0, 1.0, 1.0, "", C_U)
    ax.text(0.93, 0.70, "dy_u[0,1]\n(face length\n= Lu, dyG)", color=C_U, fontsize=9, ha="right")
    # v-face (j=1, i=0): dy_v = distance between T-points, dx_v = face length
    dim(0.2, 0.5, 0.2, 1.5, "", C_V)
    ax.text(0.26, 1.27, "dy_v[1,0]\n(T→T, dyC)", color=C_V, fontsize=9)
    dim(0.0, 1.0, 1.0, 1.0, "", C_V)
    ax.text(0.5, 0.9, "dx_v[1,0] (face length = Lv)", color=C_V, fontsize=9, ha="center")

    ax.set_xlim(-0.25, 2.55)
    ax.set_ylim(-0.25, 2.3)
    ax.set_aspect("equal")
    ax.set_xlabel("i  (x, eastward)")
    ax.set_ylabel("j  (y, northward)")
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticklabels(["i_u=0", "i_t=0", "i_u=1", "i_t=1", "i_u=2"])
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["j_v=0", "j_t=0", "j_v=1", "j_t=1", "j_v=2"])
    handles = [plt.Line2D([], [], marker="o", ls="", color=C_T, label="T-point (η, T, ρ, h): (j_t, i_t)"),
               plt.Line2D([], [], marker=">", ls="", color=C_U, label="u-point: (j_t, i_u), nx+1 columns"),
               plt.Line2D([], [], marker="^", ls="", color=C_V, label="v-point: (j_v, i_t), ny+1 rows"),
               plt.Line2D([], [], marker="s", ls="", color=C_Q, label="q-corner (f_q, ζ, PV): (j_v, i_u)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.set_title("Arakawa C grid of pyOCEAN (2 × 2 cells, south-west corner)")
    fig.tight_layout()
    fig.savefig(OUT / "cgrid_stencil.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def sigma_stretching() -> None:
    """Interfaces of 20 stretched σ-layers over a shelf/slope for several (θ_s, θ_b), plus MITgcm-delR fractions."""
    x = np.linspace(0, 60, 241)                                              # km
    H = 50 + 550 * 0.5 * (1 + np.tanh((x - 30) / 6))                        # shelf 50 m → 600 m
    cases = [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0), (6.0, 0.8)]
    fig, axes = plt.subplots(1, 5, figsize=(10, 3.8), sharey=True)
    for ax, (ts, tb) in zip(axes[:4], cases):
        ds = stretched_sigma(20, ts, tb)
        sw = np.concatenate([[0.0], np.cumsum(ds)])
        for s in sw:
            ax.plot(x, -H + s * H, color=C_U, lw=0.6)
        ax.fill_between(x, -H, -700, color="#BBBBBB")
        ax.set_title(f"θ_s = {ts:g}, θ_b = {tb:g}")
        ax.set_xlabel("x [km]")
    delr = np.array([1, 1, 2, 2, 3, 4, 5, 7, 9, 12, 15, 20, 25, 30, 40, 50, 70, 90, 120, 155.0])  # Σ = 600 m
    ds = sigma_from_thicknesses(delr)
    sw = np.concatenate([[0.0], np.cumsum(ds)])
    for s in sw:
        axes[4].plot(x, -H + s * H, color=C_V, lw=0.6)
    axes[4].fill_between(x, -H, -700, color="#BBBBBB")
    axes[4].set_title("sigma_from_thicknesses\n(delR, Σ = 600 m)")
    axes[4].set_xlabel("x [km]")
    axes[0].set_ylabel("z [m]")
    axes[0].set_ylim(-640, 5)
    fig.suptitle("σ-layer interfaces z = −H + σ_w D (η = 0), 20 layers — core.grid.sigma_levels", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "sigma_stretching.png", dpi=150)
    plt.close(fig)


def zlevel_hfac() -> None:
    """Legacy z-level grid with partial cells: wet fraction hfac_t on a 2-D section (compute_hfac)."""
    zg = create_z_grid(12, np.array([5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 25, 25.0]))  # 200 m
    x = np.arange(30)
    depth = 40 + 150 * 0.5 * (1 + np.tanh((x - 15) / 4.0))
    hf = compute_hfac(zg, depth[None, :], hfac_min=0.1)
    zf = zg["z_f"].values
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    hft = hf["hfac_t"].values[:, 0, :]
    for k in range(12):
        for i in range(30):
            if hft[k, i] > 0:
                top = zf[k]
                bot = zf[k] - hft[k, i] * (zf[k] - zf[k + 1])
                ax.add_patch(Rectangle((i, bot), 1, top - bot, facecolor=plt.cm.viridis(hft[k, i]),
                                       edgecolor="w", lw=0.4))
    ax.step(np.r_[x, 30], -np.r_[depth, depth[-1]], where="post", color="k", lw=1.2, label="bed depth(j, i)")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=ax, label="hfac_t (wet fraction)")
    ax.set_xlim(0, 30)
    ax.set_ylim(-205, 2)
    ax.set_xlabel("i_t")
    ax.set_ylabel("z [m]")
    ax.legend(loc="lower left")
    ax.set_title("Legacy z-level grid: partial cells hfac_t = min(dz, max(0, depth + z_f)) / dz, hfac_min = 0.1")
    fig.tight_layout()
    fig.savefig(OUT / "zlevel_hfac.png", dpi=150)
    plt.close(fig)


def slope_factors() -> None:
    """rx0 and rx1 (diagnostics.domain_check) on a shelf-break section, for two slope widths."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    nx, n = 40, 20
    sw = np.linspace(-1.0, 0.0, n + 1)                                       # σ_w from the bed (as check_domain)
    for col, width in enumerate((6.0, 1.5)):
        x = np.arange(nx)
        H = 50 + 550 * 0.5 * (1 + np.tanh((x - 20) / width))
        ax = axes[0, col]
        z = sw[:, None] * H[None]
        for k in range(n + 1):
            ax.plot(x + 0.5, z[k], ".-", color=C_U, lw=0.6, ms=2)
        ax.fill_between(x + 0.5, -H, -650, color="#BBBBBB")
        ax.set_title(f"tanh slope, width {width:g} cells, 20 uniform σ-layers")
        ax.set_ylabel("z [m]")
        ax.set_ylim(-650, 10)
        r0, _ = rx0(H[None, :])
        r1, _ = rx1(H[None, :], sw)
        xf = x[:-1] + 1.0
        a = axes[1, col]
        a.plot(xf, r0[: nx - 1], "o-", color=C_V, ms=3, label="rx0")
        a.set_ylabel("rx0", color=C_V)
        a.axhline(0.2, color=C_V, ls=":", lw=1)
        a2 = a.twinx()
        a2.plot(xf, r1[: nx - 1], "s-", color=C_Q, ms=3, label="rx1")
        a2.axhline(6.0, color=C_Q, ls=":", lw=1)
        a2.set_ylabel("rx1", color=C_Q)
        a.set_xlabel("u-face index (between cells i−1 and i)")
        a.legend(handles=[a.get_lines()[0], a2.get_lines()[0]], loc="upper left", frameon=False)
        a.set_title(f"max rx0 = {r0.max():.2f} (warn > 0.2),  max rx1 = {r1.max():.1f} (warn > 6)")
    fig.tight_layout()
    fig.savefig(OUT / "slope_factors.png", dpi=150)
    plt.close(fig)


def bathy_pipeline() -> None:
    """smooth_bathymetry (Hann, Shapiro) and fill_isolated_cells on a synthetic coastal bathymetry."""
    rng = np.random.default_rng(3)
    ny, nx = 40, 50
    yy, xx = np.mgrid[0:ny, 0:nx]
    coast = 10 + 3 * np.sin(yy / 5.0)
    d = np.clip(9 * (xx - coast), 0, None) + 120 * np.exp(-((xx - 32) ** 2 + (yy - 20) ** 2) / 30.0)
    d = np.where(xx > coast, d + rng.normal(0, 12, d.shape), 0.0)
    d = np.where(xx > coast, np.maximum(d, 2.0), 0.0)
    d[30, 4] = 25.0                                                          # isolated cell inside the land
    d[12, 4:int(coast[12, 0]) + 1] = 15.0                                    # 1-cell-wide inlet
    sm_h = smooth_bathymetry(d, n_passes=2, kernel="hann", n_skip=1)
    sm_s = smooth_bathymetry(d, n_passes=2, kernel="shapiro", n_skip=1)
    fi = fill_isolated_cells(sm_h)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.6), sharex=True, sharey=True)
    vmax = np.nanmax(d)
    for ax, f, t in zip(axes.ravel(), (d, sm_h, sm_s, fi),
                        ("raw depth\n(after regrid_to_model)", "smooth_bathymetry\n'hann', 2 passes, n_skip = 1",
                         "smooth_bathymetry\n'shapiro', 2 passes, n_skip = 1", "'hann' then\nfill_isolated_cells")):
        im = ax.pcolormesh(np.where(f > 0, f, np.nan), cmap="viridis", vmin=0, vmax=vmax, shading="auto")
        ax.contour(f > 0, levels=[0.5], colors="k", linewidths=0.8)
        ax.set_title(t)
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("i")
    for ax in axes[:, 0]:
        ax.set_ylabel("j")
    fig.colorbar(im, ax=axes, label="depth [m] (land white)", shrink=0.8)
    fig.savefig(OUT / "bathy_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cgrid_stencil()
    sigma_stretching()
    zlevel_hfac()
    slope_factors()
    bathy_pipeline()
    print("written:", sorted(p.name for p in OUT.glob("*.png")))
