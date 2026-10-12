"""
Test columns for the pyOCEAN ↔ MOM6 column comparisons (remapping and regridding),
written as plain text for the Fortran drivers.  Order: MOM6 convention (k = 1 at
the surface), so the drivers need no reordering; pyOCEAN calls receive the same
arrays (remapping is order-agnostic).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCHEME_NAMES = {0: "PCM", 1: "PLM", 2: "PPM_H4"}


def remap_cases() -> list[dict]:
    rng = np.random.default_rng(2026)
    cases = []
    for scheme in (0, 1, 2):
        for n0, n1 in ((10, 10), (25, 40), (50, 50), (4, 7)):
            h0 = rng.uniform(0.0, 5.0, n0)
            h0[rng.random(n0) < 0.15] = 0.0                       # vanished layers
            h0[0] = max(h0[0], 0.5)
            u0 = np.cumsum(rng.normal(0.2, 0.5, n0)) + 10.0
            h1 = rng.uniform(0.0, 5.0, n1)
            h1[0] = max(h1[0], 0.5)
            h1 *= h0.sum() / h1.sum()
            cases.append(dict(h0=h0, u0=u0, h1=h1, scheme=scheme))
    # degenerate end stencils: vanished (0), round-off (1e-10) and h_min (1e-3) layers next to thick
    # ones at both ends of the column (a direct 4×4 solve of the H4 end values is singular here)
    T = 41.36
    ends = ([0.0, T, 0.0, 0.0], [1e-3, T, 1e-3, 1e-10], [1e-10, T, 1e-3, 0.0], [0.0, 0.0, 0.0, T],
            [T, 0.0, 1e-10, 0.0], [1e-3, 1e-10, T, 0.0])
    for scheme in (0, 1, 2):
        for a in ends:
            for b in ends[::-1]:
                h0 = np.r_[a, rng.uniform(0.5, 20.0, 4), b[::-1]]
                u0 = np.cumsum(rng.normal(0.2, 0.5, h0.size)) + 10.0
                h1 = rng.uniform(0.5, 5.0, 10)
                h1 *= h0.sum() / h1.sum()
                cases.append(dict(h0=h0, u0=u0, h1=h1, scheme=scheme))
    return cases


def write_remap_cases(path: Path, cases: list[dict]) -> None:
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write(f"{SCHEME_NAMES[c['scheme']]}\n{len(c['h0'])} {len(c['h1'])}\n")
            for a in (c["h0"], c["u0"], c["h1"]):
                f.write(" ".join(f"{v:.17e}" for v in a) + "\n")


def read_remap_results(path: Path) -> list[np.ndarray]:
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    return [np.array(lines[1 + i].split(), float) for i in range(n)]


def regrid_cases() -> list[dict]:
    rng = np.random.default_rng(7)
    dz = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 70.0])
    cases = []
    for H, eta in ((201.0, 0.3), (164.0, -0.2), (40.0, 0.0), (11.0, 0.01)):
        cases.append(dict(kind="zstar", H=H, eta=eta, dz=dz))
        cases.append(dict(kind="sigma", H=H, eta=eta, dz=np.full(10, 0.1)))
    for _ in range(6):
        h = rng.uniform(0.5, 30.0, 12)
        rho = np.sort(1024.0 + rng.uniform(0.0, 4.0, 12))           # surface first, increasing
        rho[rng.integers(0, 12)] -= 0.3                              # a non-monotonic glitch is fine for interp
        targets = np.linspace(1023.5, 1028.5, 13)
        cases.append(dict(kind="interp", h=h, rho=np.sort(rho), targets=targets))
    return cases


def write_regrid_cases(path: Path, cases: list[dict]) -> None:
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write(c["kind"] + "\n")
            if c["kind"] in ("zstar", "sigma"):
                f.write(f"{len(c['dz'])} {c['H']:.17e} {c['eta']:.17e}\n")
                f.write(" ".join(f"{v:.17e}" for v in c["dz"]) + "\n")
            else:
                f.write(f"{len(c['h'])}\n")
                for a in (c["h"], c["rho"], c["targets"]):
                    f.write(" ".join(f"{v:.17e}" for v in a) + "\n")


def read_regrid_results(path: Path) -> list[np.ndarray]:
    return read_remap_results(path)
