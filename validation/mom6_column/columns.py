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


# --------------------------------------------------------------------------------------------
# Jackson et al. (2008) shear mixing (MOM6 kappa_shear_column vs physics/mixing/kappa_shear.py)
# --------------------------------------------------------------------------------------------
# Columns are stored in the **model** orientation of the oracle (k = 0 at the bed); the writer
# flips them to MOM6's surface-first order and the reader flips the results back.

KAPPA_SHEAR_PARAM_FIELDS = ("ri_crit", "shearmix_rate", "fri_curvature", "kappa_buoy_scale",
                            "tke_n_decay", "tke_shear_decay", "kd_seed", "kd_trunc_frac",
                            "tol_err", "max_kap_src_chg", "tke_background", "lz_rescale")


def _shear_layer_column(dt: float) -> dict:
    """tanh shear layer, Ri_min = 0.1, N0 = 0.01 s-1, d = 5 m, 100 × 2 m (tests/test_kappa_shear.py)."""
    n0, d, ri_min = 0.01, 5.0, 0.1
    dz = np.full(100, 2.0)
    zc = np.cumsum(dz) - 0.5 * dz
    s_max = n0 / np.sqrt(ri_min)
    u = s_max * d * np.tanh((zc - 100.0) / d)
    return dict(name=f"shear layer dt={dt:g}", dz=dz, u=u, v=np.zeros_like(u), b=n0**2 * zc,
                f2=1.0e-8, dt=dt, kappa0=1.0e-7)


def kappa_shear_cases() -> list[dict]:
    """
    Columns of the Jackson shear-mixing comparison (no thin layers, linear EOS in T: b given).

    1. tanh shear layer, Ri_min 0.1, Δt = 600 s;
    2. overflow-like bottom jet: u max 100 m above the bed on stretched layers (2–10 m),
       a dense interface (Δb 1e-3 m s-2) 30 m above the core, weak background N², a turning
       v; mixing reaches the bed (no-slip branch of the projected state);
    3. subcritical column (Ri = 4 everywhere): κ ≡ 0, Q = Q_min;
    4. the shear layer with Δt = 3600 s (more sub-steps);
    5–6. uniform shear with Ri = 0.22 (50 m, Δt 60 s) and Ri = 0.15 (200 m, Δt 3600 s): the
       whole column is turbulent, and MOM6 enters its Newton branch (4 and 2 iterations) in
       warm-started solves — see README;
    7–18. seeded random columns (N = 8–60, dz 0.5–15 m, random shear, N² ≥ 0 with noise,
       Δt ∈ {300, 1800, 7200} s, κ0 ∈ {1e-7, 1e-6, 1e-5}).
    """
    cases = [_shear_layer_column(600.0)]
    # 2. overflow-like bottom jet
    dz = np.linspace(2.0, 10.0, 70)                      # bed first: thin near the bed
    zc = np.cumsum(dz) - 0.5 * dz
    u = 0.6 * np.exp(-((zc - 100.0) / 40.0) ** 2) * (1.0 - np.exp(-zc / 15.0))
    v = 0.2 * u * np.cos(zc / 80.0)
    b = 1.0e-6 * zc + 1.0e-3 * 0.5 * (1.0 + np.tanh((zc - 130.0) / 12.0))
    cases.append(dict(name="bottom jet", dz=dz, u=u, v=v, b=b, f2=1.0e-8, dt=600.0, kappa0=1.0e-7))
    # 3. subcritical everywhere
    dz = np.full(40, 5.0)
    zc = np.cumsum(dz) - 0.5 * dz
    cases.append(dict(name="subcritical", dz=dz, u=0.005 * zc, v=np.zeros(40), b=1.0e-4 * zc,
                      f2=1.0e-8, dt=600.0, kappa0=1.0e-7))
    # 4. the shear layer with a long step
    cases.append(_shear_layer_column(3600.0))
    # 5–6. Newton cases: uniform shear S and N² = Ri S² over the whole column, so κ > 0 on
    # every interior interface and a warm-started solve (later sub-step or corrector) can stay
    # within Newton_err (MOM_kappa_shear.F90:2003-2021); see README
    for n, depth, s, ri, dt in ((30, 50.0, 0.02, 0.22, 60.0), (30, 200.0, 0.02, 0.15, 3600.0)):
        dz = np.full(n, depth / n)
        zc = np.cumsum(dz) - 0.5 * dz
        cases.append(dict(name=f"uniform shear Ri {ri} dt={dt:g} (Newton)", dz=dz, u=s * zc,
                          v=np.zeros(n), b=ri * s * s * zc, f2=1.0e-8, dt=dt, kappa0=1.0e-7))
    # 6–17. random columns
    rng = np.random.default_rng(2008)
    for i in range(12):
        n = int(rng.integers(8, 61))
        dz = rng.uniform(0.5, 15.0, n)
        zc = np.cumsum(dz) - 0.5 * dz
        n2 = np.abs(rng.normal(1.0e-5, 2.0e-5, n))
        b = np.cumsum(n2 * dz)
        u = np.cumsum(rng.normal(0.0, 0.02, n))
        v = np.cumsum(rng.normal(0.0, 0.02, n))
        cases.append(dict(name=f"random {i}", dz=dz, u=u, v=v, b=b, f2=float(rng.uniform(0.0, 2e-8)),
                          dt=float(rng.choice([300.0, 1800.0, 7200.0])),
                          kappa0=float(rng.choice([1e-7, 1e-6, 1e-5]))))
    return cases


def write_kappa_shear_cases(path: Path, cases: list[dict], params=None) -> None:
    """Write the cases for kappa_shear_driver (surface first); ``params`` = JacksonParams."""
    if params is None:
        from physics.mixing.kappa_shear import JacksonParams
        params = JacksonParams()
    reals = " ".join(f"{getattr(params, f):.17e}" for f in KAPPA_SHEAR_PARAM_FIELDS)
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write(f"{len(c['dz'])} {c['dt']:.17e} {c['f2']:.17e} {c['kappa0']:.17e}\n")
            f.write(reals + "\n")
            f.write(f"{params.max_rino_it} {params.max_ks_it}\n")
            for key in ("dz", "u", "v", "b"):
                f.write(" ".join(f"{x:.17e}" for x in np.asarray(c[key])[::-1]) + "\n")


def read_kappa_shear_results(path: Path) -> list[dict]:
    """κ and Q (interfaces, k = 0 at the bed) and the MOM6 counters of each case."""
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    out = []
    for i in range(n):
        kap, tke, cnt = lines[1 + 3 * i: 4 + 3 * i]
        nn, nr, na, ni = (int(x) for x in cnt.split())
        out.append(dict(kappa=np.array(kap.split(), float)[::-1], tke=np.array(tke.split(), float)[::-1],
                        n_newton=nn, n_ramp=nr, n_abort=na, n_iter=ni))
    return out
