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
    python eos_columns.py pgf ~/pyOCEAN_refs/build/column/pgf_driver tests/data/mom6_pgf_plm_reference.npz

The second form (task 3) writes the rows of column pairs of the nonlinear finite-volume pressure
gradient (``pgf_cases``) and the reference ``tests/data/mom6_pgf_plm_reference.npz``.

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


# ---------------------------------------------------------------------------------------------
# Nonlinear finite-volume pressure gradient (task 3): rows of columns for pgf_driver
# ---------------------------------------------------------------------------------------------

# Deep-ocean T [°C] and S profile north of the Gulf Stream (~40°N 70°W), idealised after the WOA23
# annual climatology: approximate values (the WOA OPeNDAP server was unreachable on 2026-10-31);
# the comparison needs a realistic θ–S–p range and stratification, not the exact climatology.
GULF_Z = np.array([0.0, 50.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0])
GULF_T = np.array([18.0, 15.0, 13.5, 12.0, 9.5, 7.0, 5.5, 4.8, 4.0, 3.6, 2.6, 2.3])
GULF_S = np.array([35.5, 35.6, 35.6, 35.4, 35.1, 35.0, 35.0, 34.98, 34.97, 34.96, 34.94, 34.91])
MW_MODES = ("vanished", "full", "none")      # pgf_driver mw = 1, 2, 3
EOS_TAGS = ("wright_full", "teos10")         # pgf_driver f = 1 (WRIGHT_FULL), 2 (ROQUET_RHO)
H_NV = 0.01                                  # [m] non-vanished thickness (= core.vertical.pgf.H_VANISHED)


def _gulf(z: np.ndarray, x: np.ndarray, dT: float, dS: float) -> tuple[np.ndarray, np.ndarray]:
    """Gulf profile at heights z [m] (≤ 0) with a front: T + dT·x·e^{z/800}, S + dS·x·e^{z/500}."""
    d = -np.asarray(z)
    T = np.interp(d, GULF_Z, GULF_T) + dT * x * np.exp(-d / 800.0)
    S = np.interp(d, GULF_Z, GULF_S) + dS * x * np.exp(-d / 500.0)
    return T, S


def _interfaces(H: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Interface heights (nz+1, nx) from the bed up, e₀ = −H, e_{k+1} = e_k + h_k (MOM6 order)."""
    e = np.empty((h.shape[0] + 1, h.shape[1]))
    e[0] = -H
    for k in range(h.shape[0]):
        e[k + 1] = e[k] + h[k]
    return e


def _ts_at_centres(H, h, dT, dS):
    e = _interfaces(H, h)
    x = np.linspace(-1.0, 1.0, H.size)[None, :]
    return _gulf(0.5 * (e[1:] + e[:-1]), x, dT, dS)


def pgf_cases() -> list[dict]:
    """
    Rows of columns for the PGF comparison, bed first (k = 0 at the bed), each a dict with
    ``name``, ``H`` (nx) [m], ``h``, ``T``, ``S`` (nz, nx), ``dx`` (nx−1) [m], ``rho0``,
    ``rho_ref`` [kg m⁻³], ``g`` [m s⁻²] and ``h_nv`` [m].  Every face is a column pair (L, R).

    - ``gulf_zstar``: z* layers (30 levels to 3000 m, stretched by (H+η)/H) over a slope
      3000 → 400 m with partial bed cells and 1e-10 m vanished levels below the bed; η ±0.3 m.
    - ``gulf_sigma``: 20 σ-layers over a steep slope 2000 → 150 m (bed-step slope parameter 0.6).
    - ``thin_layers``: hybrid-like columns with 1 mm and 1e-10 m layers between thick ones, in
      different positions in neighbouring columns (vanished next to non-vanished).
    - ``bottom_step``: z-levels of 24 m over the BayIW bay-mouth step 600 → 164 m with partial
      bed cells of different depths (D_L ≠ D_R) and 1e-10 m levels below the bed.
    """
    rng = np.random.default_rng(31)
    cases = []
    # z* over a slope
    nx, nz = 8, 30
    dz0 = np.geomspace(10.0, 220.0, nz)
    dz0 *= 3000.0 / dz0.sum()
    zw0 = -np.concatenate([[0.0], np.cumsum(dz0)])                      # surface first, ≤ 0
    H = np.linspace(3000.0, 400.0, nx) + np.array([0, 13.7, -21.2, 5.5, 0, -8.8, 3.3, 0])
    eta = 0.3 * np.sin(np.linspace(0.0, 2.5, nx))
    h = np.empty((nz, nx))
    for i in range(nx):
        top = zw0[:-1]
        bot = zw0[1:]
        thk = np.clip(np.minimum(top, 0.0) - np.maximum(bot, -H[i]), 0.0, None)[::-1]   # bed first
        thk = thk * (H[i] + eta[i]) / H[i]
        h[:, i] = np.where(thk > 0.0, thk, 1e-10)
    T, S = _ts_at_centres(H, h, 1.5, 0.15)
    cases.append(dict(name="gulf_zstar", H=H, h=h, T=T, S=S, dx=np.full(nx - 1, 5000.0),
                      rho0=1035.0, rho_ref=1030.0, g=9.8, h_nv=H_NV))
    # σ over a steep slope
    nx, nz = 8, 20
    H = np.array([2000.0, 1700.0, 1100.0, 600.0, 420.0, 250.0, 180.0, 150.0])
    eta = np.linspace(-0.2, 0.25, nx)
    h = np.tile((H + eta)[None, :] / nz, (nz, 1))
    T, S = _ts_at_centres(H, h, -1.0, 0.1)
    cases.append(dict(name="gulf_sigma", H=H, h=h, T=T, S=S, dx=np.linspace(2000.0, 3000.0, nx - 1),
                      rho0=1035.0, rho_ref=1035.0, g=9.8, h_nv=H_NV))
    # thin and vanished layers
    nx, nz = 6, 16
    H = np.array([800.0, 790.0, 805.0, 770.0, 760.0, 780.0])
    h = np.empty((nz, nx))
    for i in range(nx):
        thk = rng.uniform(20.0, 80.0, nz)
        thin = rng.choice(nz, 5, replace=False)
        thk[thin[:3]] = 1e-3
        thk[thin[3:]] = 1e-10
        thk *= H[i] / thk.sum()                    # keeps the 1 mm / 1e-10 m ratios to ≈ 1 %
        h[:, i] = thk
    h[:, 1] += 0.1 / nz                            # η ≈ 0.1 m in column 1
    T, S = _ts_at_centres(H, h, 2.0, 0.2)
    T += rng.normal(0.0, 0.05, T.shape)            # noise: limiter active
    cases.append(dict(name="thin_layers", H=H, h=h, T=T, S=S, dx=np.full(nx - 1, 1000.0),
                      rho0=1035.0, rho_ref=1030.0, g=9.8, h_nv=H_NV))
    # bottom step with partial cells, D_L ≠ D_R
    nx, nz = 6, 25
    H = np.array([600.0, 587.5, 600.0, 164.0, 171.2, 164.0])
    zw0 = -24.0 * np.arange(nz + 1)
    eta = np.array([0.05, -0.02, 0.0, 0.1, 0.03, -0.05])
    h = np.empty((nz, nx))
    for i in range(nx):
        thk = np.clip(np.minimum(zw0[:-1], 0.0) - np.maximum(zw0[1:], -H[i]), 0.0, None)[::-1]
        thk = thk * (H[i] + eta[i]) / H[i]
        h[:, i] = np.where(thk > 0.0, thk, 1e-10)
    T, S = _ts_at_centres(H, h, 1.0, -0.1)
    cases.append(dict(name="bottom_step", H=H, h=h, T=T, S=S, dx=np.full(nx - 1, 500.0),
                      rho0=1035.0, rho_ref=1035.0, g=9.8, h_nv=H_NV))
    return cases


def write_pgf_cases(path: Path, cases: list[dict]) -> None:
    """Input of ``pgf_driver`` (surface first; 17 significant digits)."""
    row = lambda a: " ".join(f"{v:.17e}" for v in a) + "\n"            # noqa: E731
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            nz, nx = c["h"].shape
            f.write(f"{nz} {nx}\n")
            f.write(row([c["rho0"], c["rho_ref"], c["g"], c["h_nv"]]))
            f.write(row(c["H"]))
            for name in ("h", "T", "S"):
                for k in range(nz - 1, -1, -1):
                    f.write(row(c[name][k]))
            f.write(row(c["dx"]))


def read_pgf_results(path: Path, cases: list[dict]) -> dict[str, np.ndarray]:
    """
    Parse ``pgf_driver`` output, flipped to bed first: ``c{n}_{Tt,Tb,St,Sb}`` (nz, nx) and
    ``c{n}_{eos}_{mw}_{dpa,intz,intx,pfu_mom6,pfu_bc}``; intx and the forces (nz, nx−1).
    """
    lines = iter(Path(path).read_text().split("\n"))
    assert int(next(lines)) == len(cases)
    out = {}

    def block(nz):
        return np.array([next(lines).split() for _ in range(nz)], float)[::-1].copy()

    for n, c in enumerate(cases):
        nz = c["h"].shape[0]
        for name in ("Tt", "Tb", "St", "Sb"):
            out[f"c{n}_{name}"] = block(nz)
        for tag in EOS_TAGS:
            for mw in MW_MODES:
                for name in ("dpa", "intz", "intx", "pfu_mom6", "pfu_bc"):
                    out[f"c{n}_{tag}_{mw}_{name}"] = block(nz)
    return out


def build_pgf_reference(driver: Path, npz: Path) -> None:
    """Run the compiled ``pgf_driver`` on ``pgf_cases`` and write the reference npz."""
    cases = pgf_cases()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_pgf_cases(tmp / "in.txt", cases)
        subprocess.run([str(driver), str(tmp / "in.txt"), str(tmp / "out.txt")], check=True, capture_output=True)
        res = read_pgf_results(tmp / "out.txt", cases)
    inputs = {}
    for n, c in enumerate(cases):
        for k in ("H", "h", "T", "S", "dx"):
            inputs[f"c{n}_{k}"] = c[k]
        inputs[f"c{n}_eta"] = _interfaces(c["H"], c["h"])[-1]
        inputs[f"c{n}_par"] = np.array([c["rho0"], c["rho_ref"], c["g"], c["h_nv"]])
    np.savez_compressed(npz, names=np.array([c["name"] for c in cases]), eos=np.array(EOS_TAGS),
                        mass_weight=np.array(MW_MODES), **inputs, **res)


if __name__ == "__main__":
    if sys.argv[1] == "pgf":
        build_pgf_reference(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        build_reference(Path(sys.argv[1]), Path(sys.argv[2]))
