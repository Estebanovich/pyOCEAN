"""
Figures of the manual page theory/vertical-ale.md.

Writes PNGs to docs/manual/img/vertical-ale/:
  coordinates_section.png  z*, σ, hybrid and hybrid with a shallow σ window over a shelf/slope, built by
                           core.vertical.regrid.regrid_columns (+ the initial regrid–remap passes of the model)
  ale_substeps.png         the three ALE sub-steps on a section (Lagrangian → regrid → remap), real z* regrid
                           and PPM_H4 remap
  remap_demo.png           PCM / PLM / PPM_H4 reconstructions of one column (core.vertical.remap.reconstruct)
                           and the conservative remap onto a shifted grid (remap_column)
  hybrid_column.png        one hybrid column before / after hycom1 (keep_isopycnal False and True), with a
                           stale massless bed layer
  face_clip.png            transport thickness at a u-face over a bed step (core.vertical.ale.face_transport_thickness)

Only cheap function calls (no time stepping), one thread.  Run:
    python docs/manual/figures/make_vertical_ale.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ["NUMBA_NUM_THREADS" if _v == "NUMBA_NUM_THREADS" else _v] = "1"

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from core.grid.sigma_levels import stretched_sigma  # noqa: E402
from core.vertical import regrid as rg  # noqa: E402
from core.vertical import remap as rm  # noqa: E402
from core.vertical.ale import face_transport_thickness  # noqa: E402

OUT = ROOT / "docs" / "manual" / "img" / "vertical-ale"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 9})
C_BLUE, C_ORANGE, C_GREEN, C_VERM, C_PINK, C_SKY, C_YEL = ("#0072B2", "#E69F00", "#009E73", "#D55E00",
                                                           "#CC79A7", "#56B4E9", "#000000")

RHO0, ALPHA, T0 = 1000.0, 0.2, 0.0            # SigmaBaroclinic defaults: ρ = ρ₀ − α (T − T₀)


def density(T):
    return RHO0 - ALPHA * (T - T0)


def T_profile(z):
    """Tropical-like stratification [°C]: 25 m mixed layer at 28 °C, thermocline, 5 °C at depth."""
    d = np.maximum(-z, 0.0)
    return 5.0 + 23.0 * np.where(d < 25, 1.0, np.exp(-(d - 25) / 180.0)) - 1.5 * np.tanh((d - 60) / 25.0) * (d > 25)


def cell_means(zw):
    """Exact-enough layer means of T_profile on interfaces zw (N+1, ...), by 40-point midpoint quadrature."""
    s = (np.arange(40) + 0.5) / 40
    z = zw[:-1, None] + s.reshape((1, 40) + (1,) * (zw.ndim - 1)) * (zw[1:] - zw[:-1])[:, None]
    return T_profile(z).mean(axis=1)


# ---------------------------------------------------------------- section geometry
NX = 140
X = np.linspace(0.0, 70.0, NX)                                               # km
H_SEC = np.where(X < 25, 25 + 4.0 * X, 125) + 1375 * 0.5 * (1 + np.tanh((X - 40) / 6.0))
H_SEC = H_SEC - H_SEC.min() + 25.0                                           # 25 m at the coast → 1500 m
N = 30
DZ_TOP = 1.5 * 1.12 ** np.arange(N)                                         # z* nominal Δz, surface first [m]
DZ_TOP[-1] = H_SEC.max() - DZ_TOP[:-1].sum()                                 # floor reaches the deepest bed


def hybrid_by_iterations(H, eta, params, ds, n_iter=10):
    """SigmaBaroclinic.initial_state without h_src: σ-layers, T sampled on them, n_iter regrid–remap passes."""
    wet = np.ones_like(H, bool)
    zw = rg.regrid_columns("sigma", H, eta, np.zeros((N,) + H.shape), None, {"dsigma": ds}, wet)
    h = np.diff(zw, axis=0)
    T = cell_means(zw)
    for _ in range(n_iter):
        zw = rg.regrid_columns("hybrid", H, eta, h, density(T), params, wet)
        h_new = np.diff(zw, axis=0)
        T = rm.remap_columns(h, T, h_new, rm.PPM_H4, wet)
        h = h_new
    return zw


def hybrid_from_source(H, eta, params, tgt, n_src=400):
    """SigmaBaroclinic.initial_state(h_src=...): one regrid on fine source levels (n_out = N) + snap_to_jumps."""
    wet = np.ones_like(H, bool)
    D = H + eta
    zw_src = -H[None] + D[None] * (1.0 - (1.0 - np.linspace(0.0, 1.0, n_src + 1)) ** 2)[:, None, None]
    h_src = np.diff(zw_src, axis=0)
    rho_src = density(cell_means(zw_src))
    zw = rg.regrid_columns("hybrid", H, eta, h_src, rho_src, params, wet, n_out=N)
    rg.snap_to_jumps(zw, zw_src, rho_src, float(np.min(np.abs(np.diff(tgt)))))
    return zw


def section_grids(from_source=False):
    """Interfaces (N+1, 1, NX) of z*, σ, hybrid and hybrid with a σ window."""
    H = H_SEC[None, :]
    eta = np.zeros_like(H)
    wet = np.ones_like(H, bool)
    h0 = np.zeros((N, 1, NX))
    zs = rg.regrid_columns("zstar", H, eta, h0, None, {"dz_top": DZ_TOP}, wet)
    ds = stretched_sigma(N, 3.0, 0.4)
    sg = rg.regrid_columns("sigma", H, eta, h0, None, {"dsigma": ds}, wet)
    rho_all = density(T_profile(-np.linspace(0, 1500, 600)))
    tgt = rg.target_densities(rho_all, N, n_light=4, d_rho_light=0.01)
    out = {"z*  (rule 'zstar')": zs, "σ  (rule 'sigma', θ_s = 3, θ_b = 0.4)": sg}
    for name, extra in (("hybrid  (rule 'hybrid', no σ zone)", {}),
                        ("hybrid with shallow σ window\n(d_sigma = 60 m, d_hybrid = 200 m)",
                         {"d_sigma": 60.0, "d_hybrid": 200.0})):
        params = {"dz_top": DZ_TOP, "rho_target": tgt, "dsigma": ds, "h_min": 1e-3, **extra}
        out[name] = hybrid_from_source(H, eta, params, tgt) if from_source else hybrid_by_iterations(H, eta, params, ds)
    return out, tgt


def fig_coordinates() -> None:
    grids, tgt = section_grids()
    zz = np.linspace(-1520, 0, 400)
    TT = T_profile(zz)[:, None] * np.ones(NX)[None]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharex=True, sharey=True)
    for ax, (name, zw) in zip(axes.ravel(), grids.items()):
        pc = ax.pcolormesh(X, zz, np.where(zz[:, None] > -H_SEC[None], TT, np.nan), cmap="RdYlBu_r",
                           shading="auto", vmin=5, vmax=28)
        for k in range(zw.shape[0]):
            ax.plot(X, zw[k, 0], color="k", lw=0.55)
        ax.fill_between(X, -H_SEC, -1600, color="#999999")
        ax.set_title(name)
        ax.set_ylim(-1550, 5)
        ax.set_yscale("symlog", linthresh=200, linscale=1.3)
        ax.set_yticks([0, -50, -100, -150, -200, -500, -1000, -1500])
        ax.set_yticklabels(["0", "−50", "−100", "−150", "−200", "−500", "−1000", "−1500"])
    for ax in axes[1]:
        ax.set_xlabel("x [km]")
    for ax in axes[:, 0]:
        ax.set_ylabel("z [m] (linear above −200 m)")
    fig.colorbar(pc, ax=axes, label="T [°C]  (ρ = 1000 − 0.2 T)", shrink=0.7)
    fig.suptitle(f"{N} layers on a shelf/slope section, interfaces from core.vertical.regrid.regrid_columns", y=0.98)
    fig.savefig(OUT / "coordinates_section.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_hybrid_init() -> None:
    """The two hybrid initialisations of SigmaBaroclinic.initial_state on the same section (no σ zone)."""
    H = H_SEC[None, :]
    eta = np.zeros_like(H)
    ds = stretched_sigma(N, 3.0, 0.4)
    rho_all = density(T_profile(-np.linspace(0, 1500, 600)))
    tgt = rg.target_densities(rho_all, N, n_light=4, d_rho_light=0.01)
    params = {"dz_top": DZ_TOP, "rho_target": tgt, "dsigma": ds, "h_min": 1e-3}
    a = hybrid_by_iterations(H, eta, params, ds)
    b = hybrid_from_source(H, eta, params, tgt)
    tgt0 = rg.target_densities(rho_all, N, n_light=0)
    c = hybrid_from_source(H, eta, dict(params, rho_target=tgt0), tgt0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, zw, t in ((axes[0], a, "(a) initial_state(eta, T):\nσ-layers + init_iterations = 10, n_light = 4"),
                      (axes[1], b, "(b) initial_state(eta, T, h_src): 400 levels,\nn_light = 4 → snap Δρ = 0.01 kg m⁻³"),
                      (axes[2], c, "(c) initial_state(eta, T, h_src): 400 levels,\nn_light = 0 → snap Δρ = "
                                   f"{np.min(np.abs(np.diff(tgt0))):.2f} kg m⁻³")):
        for k in range(zw.shape[0]):
            ax.plot(X, zw[k, 0], color="k", lw=0.55)
        ax.fill_between(X, -H_SEC, -400, color="#999999")
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("x [km]")
        ax.set_ylim(-300, 2)
    axes[0].set_ylabel("z [m]")
    fig.tight_layout()
    fig.savefig(OUT / "hybrid_init.png", dpi=150)
    plt.close(fig)


def fig_substeps() -> None:
    """Lagrangian interfaces displaced by an internal wave (Σh fixed per column), z* regrid, PPM_H4 remap."""
    nx, n = 16, 8
    x = np.arange(nx)
    H = np.full((1, nx), 80.0)
    eta = 0.6 * np.sin(2 * np.pi * x / nx)[None]
    dz = np.full(n, 10.0)
    zs0 = rg.regrid_columns("zstar", H, np.zeros_like(H), np.zeros((n, 1, nx)), None, {"dz_top": dz},
                            np.ones((1, nx), bool))
    # Lagrangian step: interfaces moved with the flow (mode-1 displacement), the column stretched to H + η
    zl = zs0.copy()
    sk = np.linspace(0, 1, n + 1)[:, None, None]
    zl = -H[None] + (zl + H[None]) * (H + eta)[None] / H[None]
    zl += 9.0 * np.sin(np.pi * sk) * np.cos(2 * np.pi * x / nx)[None, None]
    h_lag = np.diff(zl, axis=0)
    T_lag = cell_means(zs0)                                                  # T carried with the layers
    zt = rg.regrid_columns("zstar", H, eta, h_lag, None, {"dz_top": dz}, np.ones((1, nx), bool))
    h_t = np.diff(zt, axis=0)
    T_new = rm.remap_columns(h_lag, T_lag, h_t, rm.PPM_H4, np.ones((1, nx), bool))
    cmap, norm = plt.cm.RdYlBu_r, plt.Normalize(T_new.min(), T_new.max())
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), sharey=True)
    titles = ("1. Lagrangian step\nh† = hⁿ − Δt ∇·Q_k / A\n(no flux through interfaces)",
              "2. Regrid\nrule's target interfaces (dashed:\nz* at ηⁿ⁺¹) drawn over h†",
              "3. Remap\nT remapped conservatively\n(PPM_H4) onto hⁿ⁺¹")
    for p, ax in enumerate(axes):
        zz, TT = (zl, T_lag) if p < 2 else (zt, T_new)
        for i in range(nx):
            for k in range(n):
                ax.add_patch(Rectangle((i, zz[k, 0, i]), 1, zz[k + 1, 0, i] - zz[k, 0, i],
                                       facecolor=cmap(norm(TT[k, 0, i])), edgecolor="k", lw=0.4))
        if p == 1:
            for k in range(n + 1):
                ax.step(np.r_[x, nx], np.r_[zt[k, 0], zt[k, 0, -1]], where="post", color="k", ls="--", lw=1.2)
        ax.set_title(titles[p], fontsize=9)
        ax.set_xlim(0, nx)
        ax.set_ylim(-82, 4)
        ax.set_xlabel("i (cell)")
    axes[0].set_ylabel("z [m]")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="T [°C]", shrink=0.8)
    fig.savefig(OUT / "ale_substeps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_remap() -> None:
    """Reconstructions of one non-uniform column and the remap onto a shifted grid, for the three schemes."""
    prof = lambda z: 21.0 + 5.0 * np.tanh((z + 62.0) / 5.0) + 0.03 * z      # noqa: E731  sharp thermocline
    h = np.array([30.0, 22.0, 15.0, 9.0, 6.0, 5.0, 6.0, 9.0, 14.0, 20.0])  # bed first, Σ = 136 m
    zw = -h.sum() + np.r_[0.0, np.cumsum(h)]
    sq = (np.arange(200) + 0.5) / 200
    u = np.array([prof(zw[k] + sq * h[k]).mean() for k in range(h.size)])
    h_t = np.full(17, h.sum() / 17)                                          # target: 17 uniform layers
    zt = -h.sum() + np.r_[0.0, np.cumsum(h_t)]
    zf = np.linspace(zw[0], zw[-1], 800)

    def blocks(ax, vals, edges, **kw):
        for k, v in enumerate(vals):
            ax.plot([v, v], [edges[k], edges[k + 1]], **(kw if k == 0 else {**kw, "label": None}))

    fig, axes = plt.subplots(1, 3, figsize=(10, 5.0), sharey=True)
    for ax, (name, code) in zip(axes, (("PCM", rm.PCM), ("PLM", rm.PLM), ("PPM_H4", rm.PPM_H4))):
        coef = np.empty((h.size, 3))
        rm.reconstruct(h, u, code, rm.H_NEGLECT, rm.H_NEGLECT_EDGE, coef)
        ax.plot(prof(zf), zf, color="#777777", lw=1.0, ls=":", label="true profile")
        blocks(ax, u, zw, color="k", lw=4, alpha=0.25, label="source means (10 cells)")
        for k in range(h.size):
            xi = np.linspace(0, 1, 30)
            ax.plot(coef[k, 0] + xi * (coef[k, 1] + xi * coef[k, 2]), zw[k] + xi * h[k], color=C_BLUE, lw=1.8,
                    label="reconstruction p(ξ)" if k == 0 else None)
        out = np.empty(h_t.size)
        rm.remap_column(h, u, h_t, code, out)
        blocks(ax, out, zt, color=C_VERM, lw=1.2, marker="|", ms=6, label="remapped means (17 cells)")
        err = (out * h_t).sum() - (u * h).sum()
        ax.set_title(f"{name}\nΣ h u after − before: {err:.1e} °C m")
        ax.set_xlabel("T [°C]")
        for z in zw:
            ax.axhline(z, color="#cccccc", lw=0.5, zorder=0)
    axes[0].set_ylabel("z [m]  (grey lines: source interfaces)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("core.vertical.remap: reconstruct() and remap_column() on one column (bed and top cells stay PCM)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "remap_demo.png", dpi=150)
    plt.close(fig)


def fig_hybrid_column() -> None:
    """A hybrid column after a Lagrangian step: vanished layers, a stale light bed layer; hycom1 with/without keep."""
    n = 12
    H = np.array([[600.0]])
    eta = np.zeros((1, 1))
    dz_top = np.array([2, 3, 4, 6, 8, 12, 20, 35, 60, 100, 150, 200.0])          # z* floor, Σ = 600 m
    rho_t = np.array([1019.0, 1019.5, 1020.0, 1021.0, 1022.0, 1023.0, 1024.0, 1025.0, 1025.5, 1026.0,
                      1026.5, 1027.0, 1027.5])                                     # targets, surface first
    # source column (surface first): mixed layer, isopycnal layers near their targets, two vanished layers,
    # and a massless bed layer with a stale light density (a gravity current has covered it)
    h_top = np.array([30.0, 25.0, 40.0, 60.0, 90.0, 0.0005, 120.0, 0.0, 115.0, 0.0, 119.9995, 0.0])
    rho_top = np.array([1019.2, 1020.3, 1021.4, 1022.5, 1023.5, 1024.0, 1024.6, 1025.2, 1025.8, 1026.1,
                        1026.7, 1020.0])
    h_src = h_top[::-1].reshape(n, 1, 1)
    r_src = rho_top[::-1].reshape(n, 1, 1)
    wet = np.ones((1, 1), bool)
    res = {}
    for keep in (False, True):
        p = {"dz_top": dz_top, "rho_target": rho_t, "h_min": 1e-3, "keep_isopycnal": keep}
        zw = rg.regrid_columns("hybrid", H, eta, h_src, r_src, p, wet)
        h_new = np.diff(zw, axis=0)
        r_new = rm.remap_columns(h_src, r_src, h_new, rm.PPM_H4, wet)
        res[keep] = (zw[:, 0, 0], r_new[:, 0, 0])
    zw_src = -600.0 + np.r_[0.0, np.cumsum(h_src[:, 0, 0])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 5.2), sharey=True)
    panels = (("source after the Lagrangian step\n(● vanished layers, h < 2 h_min)", zw_src, r_src[:, 0, 0]),
              ("hycom1, keep_isopycnal = False\n(MOM6 HYCOM1: P1M_H2 everywhere)", *res[False]),
              ("hycom1, keep_isopycnal = True (default)\n(consistent interfaces stay put)", *res[True]))
    floor = -np.cumsum(dz_top)
    for ax, (title, zw, r) in zip(axes, panels):
        for k in range(n):
            ax.fill_betweenx([zw[k], zw[k + 1]], 1018.5, r[k], color=C_SKY, alpha=0.6, lw=0)
            ax.plot([r[k], r[k]], [zw[k], zw[k + 1]], color=C_BLUE, lw=2)
            if zw[k + 1] - zw[k] < 2e-3:
                ax.plot(r[k], zw[k], "o", color=C_VERM, ms=5)
        for z in zw:
            ax.axhline(z, color="k", lw=0.6)
        for t in rho_t:
            ax.axvline(t, color=C_GREEN, lw=0.6, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("ρ [kg m⁻³] (dotted: rho_target)")
        ax.set_xlim(1018.5, 1028)
    for z in floor:
        axes[1].plot([1027.6, 1028], [z, z], color=C_ORANGE, lw=2)
        axes[2].plot([1027.6, 1028], [z, z], color=C_ORANGE, lw=2)
    axes[0].set_ylabel("z [m] (black: interfaces; orange ticks: z* floor)")
    axes[0].set_ylim(-620, 0)
    fig.tight_layout()
    fig.savefig(OUT / "hybrid_column.png", dpi=150)
    plt.close(fig)


def fig_face_clip() -> None:
    """q_E (eastward, from the deep west cell) at a u-face over a bed step: z* clipped, σ not."""
    n = 10
    H = np.array([[300.0, 120.0]])
    eta = np.zeros((1, 2))
    act = np.ones((1, 2), bool)
    dz = np.full(n, 30.0)
    zs = rg.regrid_columns("zstar", H, eta, np.zeros((n, 1, 2)), None, {"dz_top": dz}, act)
    sg = rg.regrid_columns("sigma", H, eta, np.zeros((n, 1, 2)), None, {"dsigma": np.full(n, 0.1)}, act)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharey=True)
    for ax, zw, clip, name in ((axes[0], zs, True, "z* (clip = True)"), (axes[1], sg, False, "σ (clip = False)")):
        h = np.diff(zw, axis=0)
        qe, _ = face_transport_thickness(h, H, act, clip)
        for c in range(2):
            ax.hlines(zw[:, 0, c], c, c + 1, color="k", lw=0.7)
            ax.vlines([c, c + 1], -H[0, c], 0.0, color="k", lw=0.7)
        ax.fill_between([0, 1], [-300, -300], -320, color="#999999")
        ax.fill_between([1, 2], [-120, -120], -320, color="#999999")
        top_w = zw[1:, 0, 0]
        for k in range(n):
            q = qe[k, 0, 1]
            if q > 0:
                ax.add_patch(Rectangle((0.92, top_w[k] - q), 0.16, q, facecolor=C_ORANGE, edgecolor=C_VERM, lw=0.6))
            ax.text(0.5, 0.5 * (zw[k, 0, 0] + zw[k + 1, 0, 0]), f"k={k}: q_E={q:.0f} m", ha="center", va="center",
                    fontsize=8)
        ax.set_title(f"{name}: q_E at the u-face between the cells")
        ax.set_xlim(0, 2)
        ax.set_ylim(-320, 5)
        ax.set_xticks([0.5, 1.0, 1.5])
        ax.set_xticklabels(["west cell (H = 300 m)", "u-face", "east cell (H = 120 m)"])
    axes[0].set_ylabel("z [m]")
    fig.suptitle("core.vertical.ale.face_transport_thickness: orange = thickness that may cross the face eastward",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "face_clip.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_coordinates()
    fig_hybrid_init()
    fig_substeps()
    fig_remap()
    fig_hybrid_column()
    fig_face_clip()
    print("written:", sorted(p.name for p in OUT.glob("*.png")))
