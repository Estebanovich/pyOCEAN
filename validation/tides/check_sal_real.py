"""
Physical checks of the real HYCOM-IOA SAL file (tidal.salReIm_tpxo10, PAC_MEX
grid 1600×1125, from LUSTRE) against the TPXO10 ocean tide.

Self-attraction and loading is, to first order, proportional to the local
tide: η_SAL ≈ β η with β ≈ 0.08–0.12 in phase (Ray 1998; the scalar
approximation, HYCOM ``tidsal``).  HYCOM (mod_tides.F90, tidein=2) builds
etide = α η_eq − F from the file F and forces −g∇(η − etide), i.e.
−g∇(η − α η_eq + F).  The physical forcing is −g∇(η − α η_eq − η_SAL), so the
file must hold F ≈ −η_SAL ≈ −β η: the complex regression F ≈ β_F η over deep
water must give β_F ≈ −0.1 (phase ≈ 180°).  β_F ≈ +0.1 would mean the SAL
enters with the wrong sign (effective gravity g(1+β) instead of g(1−β)).

Usage:  python validation/tides/check_sal_real.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from forcing.sources.hycom_ab import read_field  # noqa: E402
from forcing.sources.tpxo import read_tpxo  # noqa: E402
from forcing.tides import CONSTITUENTS, _interp_complex, read_sal_hycom  # noqa: E402

REFS = Path.home() / "pyOCEAN_refs"
SAL_B = REFS / "tidal.salReIm_tpxo10.b"
HYCOM = Path.home() / "HYCOM"
ATLAS = REFS / "TPXO10_atlas_v2_nc"
IDM, JDM = 1600, 1125
SUB = 8                      # every 8th point (≈ 0.32°): SAL is large scale
DEEP = 1000.0                # m


def main() -> None:
    cons = list(CONSTITUENTS)
    sal = read_sal_hycom(SAL_B, IDM, JDM)                                  # (jdm, idm, nc)
    plon = read_field(HYCOM / "regional.grid.b", "plon")
    plat = read_field(HYCOM / "regional.grid.b", "plat")
    depth = read_field(HYCOM / "regional.depth_final.b", "depth", idm=IDM, jdm=JDM)
    print(f"grid lon {plon.min():.2f}…{plon.max():.2f}, lat {plat.min():.2f}…{plat.max():.2f}")
    print(f"SAL defined everywhere: {np.isfinite(sal).all()}  (HYCOM needs all points)")

    s = (slice(None, None, SUB), slice(None, None, SUB))
    lon, lat, dep, F = plon[s].ravel(), plat[s].ravel(), depth[s].ravel(), sal[s].reshape(-1, len(cons))
    tc = read_tpxo([ATLAS / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in cons], None,
                   ATLAS / "grid_tpxo10atlas_v2.nc",
                   lon_range=(lon.min() - 0.5, lon.max() + 0.5), lat_range=(lat.min() - 0.5, lat.max() + 0.5))
    eta = _interp_complex(tc.h, "lat", "lon", lat, lon)                    # (npts, nc)
    ok = np.isfinite(dep) & (dep > DEEP) & np.all(np.isfinite(eta), axis=1)
    print(f"{ok.sum()} deep-water points (> {DEEP:.0f} m)\n")

    print("con   |F|max[cm] |η|med[cm]   β_F = ΣFη*/Σ|η|²   |β_F|  arg(β_F)  corr   med|F|/|η| (|η|>10 cm)")
    for k, c in enumerate(cons):
        f, e = F[ok, k], eta[ok, k]
        beta = np.sum(f * np.conj(e)) / np.sum(np.abs(e) ** 2)
        corr = np.abs(np.sum(f * np.conj(e))) / np.sqrt(np.sum(np.abs(f) ** 2) * np.sum(np.abs(e) ** 2))
        big = np.abs(e) > 0.10
        ratio = np.median(np.abs(f[big]) / np.abs(e[big])) if big.any() else np.nan
        print(f"{c:4s}  {100 * np.abs(F[:, k]).max():8.2f}  {100 * np.median(np.abs(e)):8.2f}   "
              f"{beta.real:+.4f}{beta.imag:+.4f}i    {abs(beta):.4f}  {np.rad2deg(np.angle(beta)):+7.1f}°  "
              f"{corr:.3f}   {ratio:.3f}")


if __name__ == "__main__":
    main()
