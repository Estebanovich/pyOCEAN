"""
Figures of the manual page theory/turbulence-friction.md.

1. drag_loglaw.png      log_law_cd(D, z0) (2-D models) for several z0, first_layer_cd(Δz₁ = D/10) (σ model, 10
                        uniform layers), a constant C_d = 2.5e-3, and the effective C_d of CROCO's 3-D TIDAL_FLAT run
                        (table of validation/tidal_flat/README.md).  Real functions of physics/bottom_friction.py.
2. gls_stability.png    Canuto A stability functions c_μ(α_M, α_N) and c_μ'(α_M, α_N) built from the constants of
                        physics/mixing/gls.py (c_μ' needs CROCO's sf_nb2, which gls.py does not define), with the
                        α_M limiter of the code; the line α_N = 0 is the only one the code uses (no buoyancy).
3. kpp_profile.png      KPP shape function of physics/mixing/kpp.py (G = gat1 (3σ² − 2σ³)) against the Large et al.
                        (1994) cubic, and a sample K_s, K_m profile from obl_depth/interior_diffusivity/_blmix on one
                        wind-forced, stratified column (function calls only, no time stepping).

Run:  NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python docs/manual/figures/make_turbulence.py
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

import physics.mixing.gls as gls  # noqa: E402
from physics.bottom_friction import first_layer_cd, log_law_cd  # noqa: E402
from physics.mixing import kpp  # noqa: E402

OUT = ROOT / "docs/manual/img/turbulence"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "grey": "#666666", "black": "#000000"}


# ---------------------------------------------------------------------------------------------- 1. drag
def fig_drag() -> None:
    D = np.logspace(np.log10(0.05), np.log10(200.0), 400)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for z0, col in ((1e-5, C["sky"]), (1e-4, C["blue"]), (1e-3, C["green"]), (5e-3, C["orange"])):
        ax.plot(D, log_law_cd(D, z0), color=col, lw=2, label=f"log_law_cd, $z_0$ = {z0:g} m (2-D)")
    ax.plot(D, first_layer_cd(D / 10.0, 1e-4), color=C["blue"], ls="--", lw=1.5,
            label=r"first_layer_cd, $\Delta z_1 = D/10$, $z_0$ = 1e-4 m (σ, 10 layers)")
    ax.axhline(2.5e-3, color=C["grey"], ls=":", lw=1.5, label="constant $C_d$ = 2.5e-3")
    # effective C_d of CROCO's 3-D TIDAL_FLAT (GLS, z0 = 1e-4 m), validation/tidal_flat/README.md
    bins = [(0.2, 0.5, 2.8e-3), (1.0, 2.0, 2.1e-3), (4.0, 8.0, 1.6e-3), (12.0, 20.0, 1.3e-3)]
    for i, (a, b, cd) in enumerate(bins):
        ax.plot([a, b], [cd, cd], color=C["red"], lw=3, solid_capstyle="butt",
                label="CROCO 3-D TIDAL_FLAT, effective $C_d$" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(8e-4, 0.12)
    ax.set_xlabel("total water depth D [m]")
    ax.set_ylabel("drag coefficient $C_d$ [–]")
    ax.set_title(r"Depth-dependent drag $C_d = [\kappa/(\ln(D/z_0) - 1 + z_0/D)]^2$, clipped to [1e-4, 0.1]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "drag_loglaw.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- 2. GLS
SF_NB2 = (9.0 * gls.A1 * gls.AB3 * (gls.AB1 - gls.AB2) * gls.NN**2
          + (6.0 * gls.A1 * (gls.A2 - 3.0 * gls.A3) - 4.0 * (gls.A2**2 - 3.0 * gls.A3**2)) * gls.AB3 * gls.NN * gls.NB)


def stability(am: np.ndarray, an: float) -> tuple[np.ndarray, np.ndarray, float]:
    """c_μ, c_μ' with the code's limiters (alpha_n clipped, alpha_m ≤ am_max(alpha_n)); returns am_max too."""
    m = gls.GLSKEpsilon(2, 1)
    an = min(max(0.73 * m.alpha_n_min, an), 1e10)
    am_max = ((gls.LIM_AM0 + gls.LIM_AM1 * an + gls.LIM_AM2 * an**2 + gls.LIM_AM3 * an**3)
              / (gls.LIM_AM4 + gls.LIM_AM5 * an + gls.LIM_AM6 * an**2))
    a = np.minimum(am, am_max)
    den = (gls.SF_D0 + gls.SF_D1 * an + gls.SF_D2 * a + gls.SF_D3 * an * a + gls.SF_D4 * an**2
           + gls.SF_D5 * a**2)
    return (gls.SF_N0 + gls.SF_N1 * an + gls.SF_N2 * a) / den, (gls.SF_NB0 + gls.SF_NB1 * an + SF_NB2 * a) / den, am_max


def fig_gls() -> None:
    m = gls.GLSKEpsilon(2, 1)
    am = np.linspace(0.0, 60.0, 600)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    cases = [(0.73 * m.alpha_n_min, C["red"], "α_N = 0.73 α_N,min (convective limit)"),
             (-1.0, C["orange"], "α_N = −1 (unstable)"),
             (0.0, C["black"], "α_N = 0  ← the code (N² = 0)"),
             (5.0, C["sky"], "α_N = 5 (stable)"),
             (20.0, C["blue"], "α_N = 20 (stable)")]
    for an, col, lab in cases:
        cm, cmp_, amx = stability(am, an)
        lw = 2.6 if an == 0.0 else 1.5
        axs[0].plot(am, cm, color=col, lw=lw, label=lab)
        axs[1].plot(am, cmp_, color=col, lw=lw, ls="-" if an == 0.0 else "--")
        if amx < am.max():
            axs[0].plot([amx], [stability(np.array([amx]), an)[0][0]], "o", color=col, ms=4)
    axs[0].set_title(r"momentum $c_\mu$ (used: $\nu = c_\mu k^2/\varepsilon$)")
    axs[1].set_title(r"tracer $c_\mu'$ (CROCO; not used by pyOCEAN)")
    for ax in axs:
        ax.set_xlabel(r"$\alpha_M = (k/\varepsilon)^2 S^2$ [–]")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.16)
    axs[0].set_ylabel("stability function [–]")
    axs[0].legend(fontsize=8, loc="upper right")
    axs[0].text(0.02, 0.03, "dots: α_M clipped at α_M,max(α_N)\n(curves flat beyond)", transform=axs[0].transAxes,
                fontsize=8)
    fig.suptitle("Canuto et al. (2001) A stability functions, constants of physics/mixing/gls.py", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "gls_stability.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- 3. KPP
def fig_kpp() -> None:
    sig = np.linspace(0.0, 1.0, 200)
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.4))
    ax = axs[0]
    for gat1, col in ((0.05, C["blue"]), (0.2, C["green"])):
        ax.plot(gat1 * (3 * sig**2 - 2 * sig**3), sig, color=col, lw=2,
                label=f"pyOCEAN: $G = G_1(3σ^2 - 2σ^3)$, $G_1$ = {gat1}")
        # Large et al. (1994) eq. 11 cubic with a1 = 1, G(1) = G1, G'(1) = 0
        a2, a3 = -2.0 + 3.0 * gat1, 1.0 - 2.0 * gat1
        ax.plot(sig + a2 * sig**2 + a3 * sig**3, sig, color=col, lw=1.5, ls="--",
                label=f"Large et al. (1994): $σ + a_2σ^2 + a_3σ^3$, $G_1$ = {gat1}")
    ax.invert_yaxis()
    ax.set_xlabel("shape function G(σ) [–]")
    ax.set_ylabel(r"σ = d / $h_{bl}$ [–]")
    ax.set_title("KPP shape function (G'(1) = 0 in both)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    # sample column: 40 layers of 5 m, mixed layer 30 m over a thermocline, wind 0.1 N/m², no buoyancy flux
    nz, dzv = 40, 5.0
    dz = np.full(nz, dzv)
    z_f = -np.arange(nz + 1) * dzv
    z_t = 0.5 * (z_f[:-1] + z_f[1:])
    rho = 1025.0 + np.where(-z_t > 30.0, 0.02 * (-z_t - 30.0), 0.0)
    u_t = (0.15 * np.exp(z_t / 15.0))[:, None, None]
    v_t = np.zeros_like(u_t)
    rho3 = rho[:, None, None]
    ustar = np.full((1, 1), np.sqrt(0.1 / kpp.RHO_0))
    bflux = np.zeros((1, 1))
    hbl = kpp.obl_depth(rho3, u_t, v_t, z_t, ustar, bflux)
    km_i, ks_i = kpp.interior_diffusivity(rho3, u_t, v_t, dz)
    km, ks, _, _ = kpp._blmix(hbl, km_i, ks_i, z_f, dz, ustar, bflux, np.zeros((1, 1)), np.zeros((1, 1)))
    ax = axs[1]
    d = -z_f
    ax.semilogx(np.maximum(km[1:-1, 0, 0], 1e-7), d[1:-1], color=C["red"], lw=2, label="$K_m$ (kpp)")
    ax.semilogx(np.maximum(ks[1:-1, 0, 0], 1e-7), d[1:-1], color=C["blue"], lw=2, label="$K_s$ (kpp)")
    ax.semilogx(np.maximum(ks_i[1:-1, 0, 0], 1e-7), d[1:-1], color=C["grey"], lw=1.2, ls="--",
                label="interior $K_s$ only")
    ax.axhline(hbl[0, 0], color=C["green"], ls=":", lw=1.5, label=f"$h_{{bl}}$ = {hbl[0, 0]:.1f} m")
    ax.axhline(30.0, color=C["grey"], ls="-.", lw=1, label="top of thermocline (30 m)")
    ax.invert_yaxis()
    ax.set_xlabel("diffusivity [m² s⁻¹]")
    ax.set_ylabel("depth [m]")
    ax.set_title(r"Sample column: $\tau$ = 0.1 N m⁻², $B_f$ = 0")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "kpp_profile.png", dpi=150)
    plt.close(fig)
    print(f"KPP sample: h_bl = {hbl[0, 0]:.2f} m, max K_m = {km.max():.3e}, max K_s = {ks.max():.3e}")


if __name__ == "__main__":
    fig_drag()
    fig_gls()
    fig_kpp()
    print("written to", OUT)
