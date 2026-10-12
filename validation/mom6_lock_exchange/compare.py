"""
Hybrid ALE lock exchange: pyOCEAN against MOM6 (Ilıcak et al. 2012 configuration).

Front speeds between 2 h and 6 h, measured where the dense (or light) water column is thicker
than 1 m (nose) and 5 m (body of a 10 m Benjamin current), divided by U = ½ √(g′H), and the RPE
gain in 6 h per 500 m of channel width.  MOM6 runs: run_mom6.sh layer|rho.  pyOCEAN: the hybrid
rule of tests/test_sigma_ale.py (10 layers) with keep_isopycnal True/False and ALE-σ.

Usage: NUMBA_NUM_THREADS=5 python validation/mom6_lock_exchange/compare.py
References: Ilıcak, M. et al. (2012). Ocean Modell. 45-46, 37-58. Benjamin, T. B. (1968). JFM 31, 209-248.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from diagnostics.mixing import reference_potential_energy  # noqa: E402

RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_lock_exchange")
U = 0.5 * np.sqrt(9.8 * 5.0 / 1000.0 * 20.0)                  # Benjamin speed [m s⁻¹]
THRESH = (1.0, 5.0)


def speeds(x, dense_light):
    """dense_light: {t: (dense thickness, light thickness)} at t = 7200, 21600 s → {thr: (dense, light)} / U."""
    fr = {t: {d: (x[dl[0] > d].max() - 32e3, 32e3 - x[dl[1] > d].min()) for d in THRESH} for t, dl in dense_light.items()}
    return {d: tuple((fr[21600.0][d][k] - fr[7200.0][d][k]) / 14400 / U for k in range(2)) for d in THRESH}


def mom6(mode: str) -> dict:
    import netCDF4 as nc
    ic, d = nc.Dataset(RUNS / mode / "GOLD_IC.nc"), nc.Dataset(RUNS / mode / "prog.nc")
    t = np.asarray(d["Time"][:])
    x = np.asarray(d["xh"][:]) * 1e3 + 32e3
    area = 500.0 * 1000.0                                      # cell 0.5 km × 1 km

    def rpe(h, T):
        rho, V = 1027.0 - 0.2 * (T - 5.0), h * area
        ok = V > 0
        return reference_potential_energy(rho[ok], V[ok], area=area * h.shape[-1] * h.shape[-2])

    dl, umax = {}, 0.0
    for tt in (7200.0, 21600.0):
        i = int(np.argmin(abs(t - tt)))
        h, T = np.asarray(d["h"][i, :, 1, :]), np.asarray(d["temp"][i, :, 1, :])
        dl[tt] = (np.where(T < 17.5, h, 0).sum(0), np.where(T > 17.5, h, 0).sum(0))
        umax = max(umax, float(np.abs(d["u"][i]).max()))
    i6 = int(np.argmin(abs(t - 21600.0)))
    gain = rpe(np.asarray(d["h"][i6]), np.asarray(d["temp"][i6])) - rpe(np.asarray(ic["h"][0]), np.asarray(ic["Temp"][0]))
    rows_per_500m = 4 * 1000.0 / 500.0                         # 4 rows of 1 km = 8 widths of 500 m
    return dict(speed=speeds(x, dl), rpe=gain / rows_per_500m, umax=umax)


def pyocean(rule: str, keep: bool = True) -> dict:
    import tests.test_sigma_ale as t
    from tests.test_sigma_ale import TCOEF
    if rule == "hybrid":
        m, st = t._hybrid_lock()
        m.regrid_params["keep_isopycnal"] = keep
    else:
        m = t._ale_box("sigma", nx=128, nz=10, depth=20.0, dx=500.0, dt=20.0, params={})
        st = m.initial_state(np.zeros((1, 128)), np.where(np.arange(128) < 64, 5.0, 10.0)[None, None, :].repeat(10, 0) / TCOEF)
    r0 = m.rpe(st)
    x = (np.arange(128) + 0.5) * 500.0
    half = 7.5 / TCOEF
    dl, umax = {}, 0.0
    for n in range(1, 1081):
        st = m.step(st)
        umax = max(umax, float(np.abs(st.u).max()))
        if n * m.dt in (7200.0, 21600.0):
            dl[n * m.dt] = (np.where(st.T < half, st.h, 0).sum(0)[0], np.where(st.T > half, st.h, 0).sum(0)[0])
    return dict(speed=speeds(x, dl), rpe=m.rpe(st) - r0, umax=umax)


def main() -> None:
    rows = [("MOM6 layer (21 capas)", mom6("layer")), ("MOM6 RHO (21 capas, P1M_H4)", mom6("rho")),
            ("pyOCEAN híbrida keep_isopycnal (10)", pyocean("hybrid", True)),
            ("pyOCEAN híbrida P1M (10)", pyocean("hybrid", False)), ("pyOCEAN ALE-σ (10)", pyocean("sigma"))]
    print(f"{'caso':38s} | frente 1 m (denso, ligero) | cuerpo 5 m (denso, ligero) | RPE 6 h [J/500 m] | max|u|")
    for name, r in rows:
        s1, s5 = r["speed"][1.0], r["speed"][5.0]
        print(f"{name:38s} | {s1[0]:5.2f} {s1[1]:5.2f} U               | {s5[0]:5.2f} {s5[1]:5.2f} U               "
              f"| {r['rpe']:10.2e}        | {r['umax']:.2f}")


if __name__ == "__main__":
    main()
