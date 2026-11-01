"""
Effect of the two-tracer projection in Jackson et al. (2008) shear mixing with a nonlinear EOS (2026-11-01).

Before 2026-11-01 ``physics.mixing.kappa_shear`` carried one locally referenced buoyancy b through MOM6's
``calculate_projected_state`` (b_0 = 0, b_q = b_q−1 + dbuoy_dT ΔT + dbuoy_dS ΔS of the merged column,
``_local_buoyancy``).  MOM6 carries T and S separately with dbuoy_dT, dbuoy_dS fixed per column
(MOM_kappa_shear.F90:1126-1136, 1493-1503); pyOCEAN does too since 2026-11-01.  This script records the size of
the change on realistic columns: the WOA23 annual profile at ~40 °N 70 °W (``tests/data/woa23_gulfstream_profile.npz``,
θ and S_P; Θ and S_A for teos10 by ``physics.eos.convert``), 0–2800 m on 5 m layers to 500 m, 25 m to 1500 m and
50 m below, with a tanh shear layer (half-width max(5 m, 2 Δz)) of Ri_min 0.05, 0.1 or 0.2 at 30, 80, 150, 300, 600
or 1000 m (18 columns per EOS), f at 40 °N, κ0 = 1e-6 m² s⁻¹, Δt 600 and 3600 s, JacksonParams() defaults.

"old" = the one-buoyancy path, rebuilt from the current module: ``kappa_shear_columns_numpy`` with the linear
(b) path and b = ``_local_buoyancy`` of the column (no layer of these columns is merged; the former code took
the merged T = Σ h T / Σ h, which differs from T by an ulp).  "new" = ``kappa_shear_columns`` with T, S, eos.

Printed per EOS and Δt: max |κ_new − κ_old| / max κ_new per column (median and max over the columns) and the
relative change of the column sum of κ.

Usage:  python validation/mom6_column/kappa_shear_eos_effect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.components.closure import thin_layer_threshold  # noqa: E402
from physics.eos.convert import sp_to_sa, theta_to_ct  # noqa: E402
from physics.eos.stratification import eos_code, interface_pressure, n2_local  # noqa: E402
from physics.mixing import kappa_shear as ks  # noqa: E402

RHO0, G = 1035.0, 9.81
DEPTHS = (30.0, 80.0, 150.0, 300.0, 600.0, 1000.0)     # [m] centre of the shear layer
RIS = (0.05, 0.1, 0.2)                                 # Ri_min at the centre [nondim]


def woa_columns() -> tuple[np.ndarray, np.ndarray, dict]:
    """Layer thicknesses (bed first) [m], layer-centre depths [m] and {eos: (T, S)} of the WOA23 column."""
    d = np.load(ROOT / "tests" / "data" / "woa23_gulfstream_profile.npz")
    edges = np.concatenate([np.arange(0.0, 500.0, 5.0), np.arange(500.0, 1500.0, 25.0),
                            np.arange(1500.0, 2801.0, 50.0)])
    zc = 0.5 * (edges[1:] + edges[:-1])
    th = np.interp(zc, d["depth"], d["theta"])[::-1].copy()
    sp = np.interp(zc, d["depth"], d["SP"])[::-1].copy()
    sa = sp_to_sa(sp)
    return np.diff(edges)[::-1].copy(), zc[::-1].copy(), {"wright_full": (th, sp), "teos10": (theta_to_ct(sa, th), sa)}


def main() -> None:
    dz, depth, ts = woa_columns()
    n = dz.size
    cols = [(z, ri) for z in DEPTHS for ri in RIS]
    m = len(cols)
    p = ks.JacksonParams()
    DZ = np.tile(dz[:, None], (1, m))
    thr = thin_layer_threshold(DZ)
    f2 = np.full(m, (2.0 * 7.2921e-5 * np.sin(np.radians(40.0))) ** 2)
    iface_depth = 0.5 * (depth[1:] + depth[:-1])
    for eos, (T, S) in ts.items():
        n2 = n2_local(T[:, None], S[:, None], dz[:, None], interface_pressure(dz[:, None], G, RHO0), eos,
                      rho0=RHO0, g=G)[:, 0]
        U = np.zeros((n, m))
        for j, (zs, ri) in enumerate(cols):
            w = max(5.0, 2.0 * np.interp(zs, depth[::-1], dz[::-1]))
            k = int(np.argmin(np.abs(iface_depth - zs)))
            U[:, j] = np.sqrt(max(n2[k], 1e-8) / ri) * w * np.tanh((zs - depth) / w)
        Tm, Sm = np.tile(T[:, None], (1, m)), np.tile(S[:, None], (1, m))
        hb = np.zeros(n)
        ks._local_buoyancy(T.copy(), S.copy(), dz.copy(), n, eos_code(eos), RHO0, G, hb)
        B = np.tile(hb[:, None], (1, m))
        for dt in (600.0, 3600.0):
            k_new, _ = ks.kappa_shear_columns(U, 0.0 * U, np.zeros_like(U), DZ, f2, dt, 1e-6, thr, p, T=Tm, S=Sm,
                                              eos=eos, rho0=RHO0, g=G)
            k_old, _ = ks.kappa_shear_columns_numpy(U, 0.0 * U, B, DZ, f2, dt, 1e-6, thr, p)
            rel = np.abs(k_new - k_old).max(axis=0) / k_new.max(axis=0)
            rsum = np.abs(k_new.sum(axis=0) - k_old.sum(axis=0)) / k_new.sum(axis=0)
            print(f"{eos:11s} dt {dt:5.0f} s  max|dk|/max k: median {np.median(rel):.1e}, max {rel.max():.1e}"
                  f" (z {cols[int(rel.argmax())][0]:.0f} m, Ri {cols[int(rel.argmax())][1]});"
                  f"  |d sum k|/sum k: median {np.median(rsum):.1e}, max {rsum.max():.1e};"
                  f"  max k {k_new.max(axis=0).min():.1e}-{k_new.max(axis=0).max():.1e} m2/s")


if __name__ == "__main__":
    main()
