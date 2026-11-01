"""
Point cloud for the pyOCEAN ↔ MOM6 comparison of the nonlinear equations of state
(WRIGHT_FULL and ROQUET_RHO; TEOS-10 in the 3-D core, task 1), and the writer of the
committed reference table ``tests/data/mom6_eos_reference.npz``.

The cloud covers the ocean range of the 3-D model: T −2…32 °C, S 0…42 (PSU for Wright,
g/kg for Roquet), p 0…6e7 Pa (≈ 6000 dbar), 2000 uniform random points (seed 2026),
the 8 corners of that box, and 40 fresh-water points (S = 0, T 0…30 °C, p 0…1e7 Pa;
rivers, review focus 1).  The same (T, S, p) feed both EOS.

Usage (called by ``build_reference.sh`` after the driver is built)::

    python eos_columns.py ~/pyOCEAN_refs/build/column/eos_driver tests/data/mom6_eos_reference.npz

The npz holds the inputs ``T, S, p``, ``rho_ref`` (1035 kg m⁻³) and, for
``w`` = WRIGHT_FULL and ``r`` = ROQUET_RHO, ``{w,r}_rho`` [kg m⁻³], ``{w,r}_rho_anom``
(ρ − ρ_ref in MOM6's anomaly form) [kg m⁻³], ``{w,r}_drho_dT`` [kg m⁻³ °C⁻¹] and
``{w,r}_drho_dS`` [kg m⁻³ (PSU or g/kg)⁻¹]; plus the compiled coefficients as
``coef_names`` / ``coef_values`` (``eos_driver coef``).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RHO_REF = 1035.0     # reference density of the anomaly outputs [kg m-3] (NWA12 RHO_PGF_REF)


def eos_cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T [°C], S [PSU or g/kg], p [Pa]) of the comparison cloud, 2048 points."""
    rng = np.random.default_rng(2026)
    n = 2000
    T = rng.uniform(-2.0, 32.0, n)
    S = rng.uniform(0.0, 42.0, n)
    p = rng.uniform(0.0, 6.0e7, n)
    corners = np.array([(t, s, q) for t in (-2.0, 32.0) for s in (0.0, 42.0) for q in (0.0, 6.0e7)])
    fresh = np.column_stack([np.linspace(0.0, 30.0, 40), np.zeros(40),
                             np.tile(np.linspace(0.0, 1.0e7, 8), 5)])
    pts = np.vstack([np.column_stack([T, S, p]), corners, fresh])
    return pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()


def write_eos_cloud(path: Path, T: np.ndarray, S: np.ndarray, p: np.ndarray) -> None:
    """Plain-text input of ``eos_driver eval`` (17 significant digits: exact round trip)."""
    with open(path, "w") as f:
        f.write(f"{T.size}\n")
        for t, s, q in zip(T, S, p):
            f.write(f"{t:.17e} {s:.17e} {q:.17e}\n")


def read_eos_results(path: Path) -> dict[str, np.ndarray]:
    """Parse ``eos_driver eval`` output into ``{w,r}_{rho,rho_anom,drho_dT,drho_dS}``."""
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    out = {}
    for tag, start in (("w", 1), ("r", 1 + n)):
        a = np.array([lines[start + i].split() for i in range(n)], float)
        for j, name in enumerate(("rho", "rho_anom", "drho_dT", "drho_dS")):
            out[f"{tag}_{name}"] = a[:, j].copy()
    return out


def read_coefficients(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``eos_driver coef`` output into (names, values)."""
    names, values = [], []
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            k, v = line.split()
            names.append(k)
            values.append(float(v))
    return np.array(names), np.array(values)


def build_reference(driver: Path, npz: Path) -> None:
    """Run the compiled driver on the cloud and write the reference npz."""
    T, S, p = eos_cloud()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_eos_cloud(tmp / "in.txt", T, S, p)
        subprocess.run([str(driver), "eval", str(tmp / "in.txt"), str(tmp / "out.txt")],
                       check=True, capture_output=True)
        subprocess.run([str(driver), "coef", str(tmp / "coef.txt")], check=True, capture_output=True)
        res = read_eos_results(tmp / "out.txt")
        names, values = read_coefficients(tmp / "coef.txt")
    np.savez_compressed(npz, T=T, S=S, p=p, rho_ref=RHO_REF, coef_names=names, coef_values=values, **res)


if __name__ == "__main__":
    build_reference(Path(sys.argv[1]), Path(sys.argv[2]))
