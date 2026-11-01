"""
Volume-weighted T-S diagram of a model state (TEOS-10 in the 3-D core, task 8).

``ts_histogram`` bins the (T, S) of the cells by the volume of water they hold; ``plot_ts`` draws it with contours of
the potential density σ of the **same equation of state** as the model, so the lines are the ones the model's
pressure gradient sees (Θ, S_A for ``teos10``; θ, S_P for ``linear`` and ``wright_full``).  T and S are in the EOS
convention of the model (``eos.ts_kind``), i.e. the prognostic fields of ``SigmaState``; use
``physics.eos.convert.from_model_ts`` first for a θ, S_P diagram of a ``teos10`` state.

Index order of the inputs is free (any shape; T, S and volume must have the same one).  matplotlib is imported
inside ``plot_ts`` only, so the core never needs it.
"""
from __future__ import annotations

import numpy as np


def ts_histogram(T: np.ndarray, S: np.ndarray, volume: np.ndarray, t_bins: np.ndarray, s_bins: np.ndarray) -> np.ndarray:
    """
    Volume [m³] of water in each (T, S) bin.

    Parameters
    ----------
    T, S : arrays of one shape; temperature [°C] and salinity (PSU or g kg⁻¹) of each cell.
    volume : array of the same shape; cell volume [m³] (dry and land cells: 0 or NaN).
    t_bins, s_bins : bin edges, increasing (len n_t + 1, n_s + 1).

    Returns
    -------
    (n_t, n_s) array: the sum of ``volume`` of the cells in each bin.  Cells with a non-finite T, S or volume are
    dropped, as are those outside the bins, so ``hist.sum()`` is the volume inside the bins (the total volume when
    the bins cover the data).
    """
    T, S, V = (np.asarray(a, float).ravel() for a in (T, S, volume))
    if not (T.shape == S.shape == V.shape):
        raise ValueError(f"T {T.shape}, S {S.shape} and volume {V.shape} must have the same size")
    ok = np.isfinite(T) & np.isfinite(S) & np.isfinite(V)
    hist, _, _ = np.histogram2d(T[ok], S[ok], bins=(np.asarray(t_bins, float), np.asarray(s_bins, float)),
                                weights=V[ok])
    return hist


def plot_ts(T: np.ndarray, S: np.ndarray, volume: np.ndarray, eos, p_ref: float = 0.0, ax=None, n_bins: int = 80):
    """
    Volume-weighted T-S diagram with σ contours of ``eos`` at the reference pressure ``p_ref`` [Pa] (0 = σ₀,
    2e7 = σ₂): the contours are ``eos.potential_density(T, S, p_ref) − 1000`` [kg m⁻³] on the plot grid.

    Parameters
    ----------
    T, S, volume : arrays of one shape; T, S in the convention of ``eos`` (``eos.ts_kind``).
    eos : an EOS of ``core.components.eos`` (``make_eos(...)`` or ``model._eos``).
    p_ref : reference pressure of the potential density [Pa].
    ax : matplotlib axes (a new figure when None).
    n_bins : bins per axis, spanning the finite data.

    Returns
    -------
    The matplotlib axes.
    """
    import matplotlib.pyplot as plt                              # only here: the core does not need matplotlib

    T, S, V = (np.asarray(a, float).ravel() for a in (T, S, volume))
    ok = np.isfinite(T) & np.isfinite(S) & np.isfinite(V) & (V > 0)
    if not ok.any():
        raise ValueError("no wet cell with finite T, S and volume to plot")
    t_bins = np.linspace(T[ok].min(), T[ok].max() + 1e-9, n_bins + 1)
    s_bins = np.linspace(S[ok].min(), S[ok].max() + 1e-9, n_bins + 1)
    hist = ts_histogram(T, S, V, t_bins, s_bins)
    if ax is None:
        _, ax = plt.subplots()
    vol = np.where(hist > 0, hist, np.nan)
    mesh = ax.pcolormesh(s_bins, t_bins, vol, norm=_log_norm(vol), cmap="viridis")
    ax.figure.colorbar(mesh, ax=ax, label="volume [m³]")
    # σ contours of the model's own EOS over a margin around the data
    ts = np.linspace(t_bins[0], t_bins[-1], 60)
    ss = np.linspace(s_bins[0], s_bins[-1], 60)
    Sg, Tg = np.meshgrid(ss, ts)
    sigma = eos.potential_density(Tg, Sg, np.full_like(Tg, p_ref)) - 1000.0
    cs = ax.contour(Sg, Tg, sigma, colors="k", linewidths=0.6)
    ax.clabel(cs, fmt="%.1f", fontsize=7)
    cons = eos.ts_kind == "ct_sa"
    ax.set_xlabel("S_A [g kg⁻¹]" if cons else "S_P")
    ax.set_ylabel("Θ [°C]" if cons else "θ [°C]")
    ax.set_title(f"T-S (eos={eos.name}), σ at p_ref = {p_ref:g} Pa")
    return ax


def _log_norm(vol: np.ndarray):
    from matplotlib.colors import LogNorm
    return LogNorm(vmin=np.nanmin(vol), vmax=np.nanmax(vol))
