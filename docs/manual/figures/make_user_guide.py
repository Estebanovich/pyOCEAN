#!/usr/bin/env python
"""
Figures of the user-guide chapter (docs/manual/user-guide/output-restart-diagnostics.md).

Runs the σ lock exchange of validation/lock_exchange (128 × 20 layers, Δt = 10 s, 6 h), writes CF output with
diagnostics.output.OutputWriter (NetCDF4, one record per hour) and the spurious-mixing diagnostics (RPE, Σ mix_num),
then reads the file back with xarray exactly as the manual shows and plots:

  img/user-guide/output_lock_exchange.png : T section at 6 h on the σ heights z = η + s_rho (h + η), and the
                                            hourly RPE change and Σ mix_num.

One thread (NUMBA_NUM_THREADS = 1, BLAS/OMP = 1); ~1 min on one core.
Usage: /opt/anaconda3/bin/python docs/manual/figures/make_user_guide.py
"""
from __future__ import annotations

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
import tempfile  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
warnings.filterwarnings("ignore", message="The input coordinates to pcolormesh")   # σ heights: monotone in k only

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, set_threads  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402
from diagnostics.output import OutputWriter  # noqa: E402

OUT = ROOT / "docs" / "manual" / "img" / "user-guide"
L, H, NX, NZ, DT, HOURS = 64e3, 20.0, 128, 20, 10.0, 6
RHO0, TCOEF = 1000.0, 0.28


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    set_threads(1)
    hg = create_cartesian_grid(nx=NX, ny=1, dx=L / NX, dy=L / NX, f0=0.0)
    bathy = np.full((1, NX), H)
    m = SigmaBaroclinic(hg, bathy, dt=DT, n_layers=NZ, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-6, ah=0.0)
    T0 = np.where(np.arange(NX) < NX // 2, 5.0 / TCOEF, 10.0 / TCOEF)
    st = m.initial_state(np.zeros((1, NX)), np.broadcast_to(T0, (NZ, 1, NX)).copy())
    rpe0 = m.rpe(st)
    t_h, drpe, mix = [0.0], [0.0], [0.0]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lock.nc"
        w = OutputWriter(path, hg, bathy, n_layers=NZ, dsigma=m.core.dsigma, fmt="netcdf", model=m,
                         start="2026-01-01T00:00:00", title="lock exchange (manual figure)")
        w.write(st.time, eta=st.eta, u=st.u, v=st.v, T=st.T)
        every = int(round(3600 / DT))
        for n in range(1, int(round(HOURS * 3600 / DT)) + 1):
            st = m.step(st)
            if n % every == 0:
                w.write(st.time, eta=st.eta, u=st.u, v=st.v, T=st.T, mix_num=st.mix_num)
                t_h.append(st.time / 3600)
                drpe.append((m.rpe(st) - rpe0) / rpe0)
                mix.append(float(st.mix_num.sum()))
        w.close()

        # --- reading back, as in the manual
        ds = xr.open_dataset(path)
        last = ds.isel(time=-1, yh=0)
        z = (last.eta + ds.s_rho * (ds.h.isel(yh=0) + last.eta)).transpose("s_rho", "xh")   # CF ocean_sigma_coordinate
        x = np.broadcast_to(ds.xh.values / 1e3, z.shape)
        fig, ax = plt.subplots(2, 1, figsize=(8, 6.2), gridspec_kw=dict(height_ratios=[1.3, 1]))
        pc = ax[0].pcolormesh(x, z.values, (last["T"].values * TCOEF), cmap="RdBu_r", shading="nearest")
        ax[0].set_xlabel("x [km]")
        ax[0].set_ylabel("z [m]")
        ax[0].set_title(f"Lock exchange, σ model: α T at {HOURS} h, read from the CF output", fontsize=10)
        fig.colorbar(pc, ax=ax[0], label="α T [kg m⁻³]")
        ds.close()
    a2 = ax[1]
    a2.plot(t_h, drpe, "o-", color="#0072B2", label="(RPE − RPE₀)/RPE₀  (model.rpe)")
    a2.set_xlabel("time [h]")
    a2.set_ylabel("(RPE − RPE₀)/RPE₀", color="#0072B2")
    b2 = a2.twinx()
    b2.plot(t_h, mix, "s--", color="#D55E00", label="Σ mix_num")
    b2.set_ylabel("Σ mix_num [°C² m³ s⁻¹]", color="#D55E00")
    a2.set_title("Spurious-mixing diagnostics, hourly", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "output_lock_exchange.png", dpi=150)
    print("wrote", OUT / "output_lock_exchange.png", "dRPE/RPE0 at end", drpe[-1])


if __name__ == "__main__":
    main()
