"""
Figures of the manual page theory/free-surface-time-stepping.md (analytic, no model run).

1. ab_amplification.png  Amplification factor |λ| of the physical root for the oscillation equation
                         dy/dt = iωy with forward Euler, AB2 (3/2, −1/2), AB3 (23/12, −16/12, 5/12) and
                         SSP-RK3 (the tracer scheme); AB3 stability limit on the imaginary axis.
2. theta_damping.png     θ-method for the same oscillation (the free-surface / transport weighting):
                         |λ| per step and amplitude left after one period, θ = 0.5, 0.6, 1, with the
                         small-ωΔt estimate 1 − (θ − ½)(ωΔt)².

Run:  python docs/manual/figures/make_time_stepping.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/manual/img/free-surface-time-stepping"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})


def physical_root(coeffs_fn, w):
    """Root of the characteristic polynomial closest to exp(iωΔt)."""
    lam = np.roots(coeffs_fn(1j * w))
    return lam[np.argmin(np.abs(lam - np.exp(1j * w)))]


def ab2_poly(z):          # λ² − λ − z(3/2 λ − 1/2) = 0
    return [1.0, -1.0 - 1.5 * z, 0.5 * z]


def ab3_poly(z):          # λ³ − λ² − z(23/12 λ² − 16/12 λ + 5/12) = 0
    return [1.0, -1.0 - 23.0 / 12.0 * z, 16.0 / 12.0 * z, -5.0 / 12.0 * z]


def max_root(coeffs_fn, w):
    return np.abs(np.roots(coeffs_fn(1j * w))).max()


def ab_figure() -> None:
    w = np.linspace(1e-3, 1.0, 400)
    fe = np.sqrt(1 + w**2)
    ab2 = np.array([max_root(ab2_poly, x) for x in w])
    ab3 = np.array([max_root(ab3_poly, x) for x in w])
    z = 1j * w
    rk3 = np.abs(1 + z + z**2 / 2 + z**3 / 6)
    # AB3 limit on the imaginary axis
    lim = w[np.argmax(ab3 > 1 + 1e-9)] if np.any(ab3 > 1 + 1e-9) else np.nan
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    ax = axes[0]
    ax.plot(w, fe, color="#666666", label="forward Euler (first step)")
    ax.plot(w, ab2, color="#E69F00", label="AB2 (second step; legacy model)")
    ax.plot(w, ab3, color="#0072B2", lw=2, label="AB3 (momentum, from the third step)")
    ax.plot(w, rk3, color="#009E73", ls="--", label="SSP-RK3 (tracers)")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.axvline(lim, color="#0072B2", lw=0.8, ls=":")
    ax.text(lim + 0.015, 0.94, f"AB3 limit\n ωΔt ≈ {lim:.3f}", color="#0072B2", fontsize=9)
    ax.set_xlabel(r"$\omega\Delta t$")
    ax.set_ylabel(r"largest $|\lambda|$ per step")
    ax.set_ylim(0.9, 1.12)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8.5)
    ax.set_title(r"(a) $dy/dt = i\omega y$: amplification per step")
    ax = axes[1]
    per = 2 * np.pi / w                                    # steps per period
    ax.semilogy(w, np.abs(ab2 ** per - 1) + 1e-16, color="#E69F00", label="AB2: growth per period")
    ab3_phys = np.array([np.abs(physical_root(ab3_poly, x)) for x in w])
    ax.semilogy(w, np.abs(ab3_phys ** per - 1) + 1e-16, color="#0072B2", lw=2, label="AB3: damping per period")
    ax.semilogy(w, np.abs(rk3 ** per - 1) + 1e-16, color="#009E73", ls="--", label="SSP-RK3: damping per period")
    ax.set_xlabel(r"$\omega\Delta t$")
    ax.set_ylabel(r"$\left|\,|\lambda|^{2\pi/\omega\Delta t} - 1\right|$")
    ax.set_xlim(0, lim)
    ax.set_ylim(1e-6, 1)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("(b) amplitude change per oscillation period")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(OUT / "ab_amplification.png", dpi=150)
    plt.close(fig)
    print(f"AB3 stability limit on the imaginary axis: ωΔt = {lim:.4f}")


def theta_figure() -> None:
    w = np.linspace(1e-3, 2.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    cols = {0.5: "#009E73", 0.6: "#0072B2", 1.0: "#D55E00"}
    for th, col in cols.items():
        lam = np.abs((1 + 1j * (1 - th) * w) / (1 - 1j * th * w))
        axes[0].plot(w, lam, color=col, lw=2, label=f"θ = {th}")
        if th != 0.5:
            axes[0].plot(w, 1 - (th - 0.5) * w**2, color=col, ls=":", lw=1)
        per = 2 * np.pi / w
        axes[1].plot(w, lam ** per, color=col, lw=2, label=f"θ = {th}")
    axes[0].set_xlabel(r"$\omega\Delta t$")
    axes[0].set_ylabel(r"$|\lambda|$ per step")
    axes[0].set_ylim(0.4, 1.02)
    axes[0].legend(fontsize=9)
    axes[0].set_title(r"(a) θ-method: $|\lambda| = |1+i(1-\theta)\omega\Delta t|\,/\,|1-i\theta\omega\Delta t|$"
                      "\n(dotted: $1-(\\theta-\\frac{1}{2})(\\omega\\Delta t)^2$)")
    axes[1].set_xlabel(r"$\omega\Delta t$")
    axes[1].set_ylabel("amplitude left after one period")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=9)
    axes[1].set_title("(b) damping of one oscillation period")
    fig.tight_layout()
    fig.savefig(OUT / "theta_damping.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ab_figure()
    theta_figure()
    print("written:", sorted(p.name for p in OUT.glob("*.png")))
