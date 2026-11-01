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


# Jackson shear mixing with a nonlinear EOS (two-tracer projection, 2026-11-01): MOM6 carries T and S
# through calculate_projected_state with dbuoy_dT, dbuoy_dS fixed per column (MOM_kappa_shear.F90:1126-1136,
# 1493-1503); ``kappa_shear_driver eos``.  Columns bed first; T, S in the EOS convention (θ, S_P for
# wright_full; Θ, S_A for teos10).
KAPPA_SHEAR_EOS_FORMS = {"wright_full": 4, "teos10": 7}      # MOM_EOS.F90:161-168 (teos10 = ROQUET_RHO)
SA_SP = 35.16504 / 35.0                                      # S_A / S_P (TEOS-10 reference composition)


def _scaled_shear(T: np.ndarray, S: np.ndarray, dz: np.ndarray, eos: str, ri: float, profile: np.ndarray,
                  rho0: float, g: float) -> np.ndarray:
    """u = s · profile with s chosen so that the largest locally referenced N² of (T, S) has Ri = ri when the
    shear is d(profile)/dz = 1 (uniform-shear Newton cases): s = √(max N² / ri)."""
    from physics.eos.stratification import interface_pressure, n2_local
    n2 = n2_local(T[:, None], S[:, None], dz[:, None], interface_pressure(dz[:, None], g, rho0), eos,
                  rho0=rho0, g=g)
    return np.sqrt(n2.max() / ri) * profile


def kappa_shear_eos_cases() -> list[dict]:
    """
    Columns of the nonlinear-EOS Jackson comparison, each for ``wright_full`` and ``teos10`` (ρ₀ 1035, g 9.81).

    1. tanh shear layer (Ri_min ≈ 0.1 at the centre) on a 200 m thermocline-halocline (2 m layers), Δt 600 s;
    2. the same with Δt 3600 s (more sub-steps);
    3. overflow bottom jet: 2000 m column (stretched layers, 4 m at the bed to 120 m at the top), cold salty
       plume 60 m thick under a warmer fresher interior, u max 0.6 m s⁻¹ 40 m above the bed (pressures ~2e7 Pa);
    4–5. uniform shear on 30 layers, Ri ≤ 0.22 (50 m, Δt 60 s) and Ri ≤ 0.15 (200 m, Δt 3600 s), T and S
       both stratified: every interior interface turbulent; MOM6 enters its Newton branch in the first (3
       iterations with either EOS), not in the second (unlike its linear counterpart: the T, S column takes
       another branch of the Picard iteration, 63 / 50 truncation-ramp events);
    6. cold fresh polar column (T −1.5 … 1 °C, salinity-stratified halocline, α small and pressure-dependent)
       with a shear layer in the halocline;
    7–14. seeded random columns (N = 8–50, dz 0.5–15 m, T and S random walks, random shear, Δt ∈
       {300, 1800, 7200} s, κ0 ∈ {1e-7, 1e-6, 1e-5}).
    """
    from physics.eos.stratification import interface_pressure, n2_local
    rho0, g = 1035.0, 9.81
    base = []
    # 1–2. shear layer on a thermocline
    dz = np.full(100, 2.0)
    zc = np.cumsum(dz) - 0.5 * dz
    T = 8.0 + 0.05 * zc
    S = 34.6 + 0.002 * zc                    # saltier upward (destabilising) but T dominates
    d, ri_min = 5.0, 0.1
    for eos in KAPPA_SHEAR_EOS_FORMS:
        Te, Se = T, (S * SA_SP if eos == "teos10" else S)
        n2 = n2_local(Te[:, None], Se[:, None], dz[:, None], interface_pressure(dz[:, None], g, rho0), eos,
                      rho0=rho0, g=g)
        n0 = float(np.sqrt(n2[49, 0]))
        s_max = n0 / np.sqrt(ri_min)
        u = s_max * d * np.tanh((zc - 100.0) / d)
        for dt in (600.0, 3600.0):
            base.append(dict(name=f"{eos} shear layer dt={dt:g}", eos=eos, dz=dz, u=u, v=np.zeros_like(u),
                             T=Te, S=Se, f2=1.0e-8, dt=dt, kappa0=1.0e-7, rho0=rho0, g=g))
    # 3. overflow bottom jet, deep column
    dz = np.geomspace(4.0, 120.0, 40)
    dz = dz * (2000.0 / dz.sum())
    zc = np.cumsum(dz) - 0.5 * dz
    plume = 0.5 * (1.0 - np.tanh((zc - 60.0) / 10.0))
    T = 3.0 + 0.004 * zc - 1.5 * plume
    S = 34.95 - 0.0002 * zc + 0.08 * plume
    u = 0.6 * np.exp(-((zc - 40.0) / 30.0) ** 2) * (1.0 - np.exp(-zc / 8.0))
    v = 0.25 * u * np.cos(zc / 50.0)
    for eos in KAPPA_SHEAR_EOS_FORMS:
        base.append(dict(name=f"{eos} deep bottom jet", eos=eos, dz=dz, u=u, v=v, T=T,
                         S=S * SA_SP if eos == "teos10" else S, f2=1.0e-8, dt=600.0, kappa0=1.0e-7,
                         rho0=rho0, g=g))
    # 4–5. Newton cases: uniform shear, Ri ≤ ri at every interface
    for n, depth, ri, dt, tag in ((30, 50.0, 0.22, 60.0, " (Newton)"), (30, 200.0, 0.15, 3600.0, "")):
        dz = np.full(n, depth / n)
        zc = np.cumsum(dz) - 0.5 * dz
        T = 10.0 + 0.04 * zc
        S = 35.2 - 0.003 * zc
        for eos in KAPPA_SHEAR_EOS_FORMS:
            Se = S * SA_SP if eos == "teos10" else S
            u = _scaled_shear(T, Se, dz, eos, ri, zc, rho0, g)
            base.append(dict(name=f"{eos} uniform shear Ri {ri} dt={dt:g}{tag}", eos=eos, dz=dz, u=u,
                             v=np.zeros(n), T=T, S=Se, f2=1.0e-8, dt=dt, kappa0=1.0e-7, rho0=rho0, g=g))
    # 6. cold fresh polar column, salinity-stratified
    dz = np.full(60, 2.5)
    zc = np.cumsum(dz) - 0.5 * dz
    T = -1.5 + 2.5 * 0.5 * (1.0 + np.tanh((zc - 75.0) / 20.0))
    S = 34.5 - 2.0 * 0.5 * (1.0 + np.tanh((zc - 100.0) / 15.0))
    u = 0.4 * np.tanh((zc - 100.0) / 6.0)
    for eos in KAPPA_SHEAR_EOS_FORMS:
        base.append(dict(name=f"{eos} polar halocline", eos=eos, dz=dz, u=u, v=0.3 * u, T=T,
                         S=S * SA_SP if eos == "teos10" else S, f2=1.9e-8, dt=1800.0, kappa0=1.0e-6,
                         rho0=rho0, g=g))
    # 7–14. random columns
    rng = np.random.default_rng(20261101)
    for i in range(8):
        n = int(rng.integers(8, 51))
        dz = rng.uniform(0.5, 15.0, n)
        T = 4.0 + np.cumsum(np.abs(rng.normal(0.02, 0.04, n)) * dz)
        S = 34.0 + np.cumsum(rng.normal(0.0, 0.002, n) * dz)
        u = np.cumsum(rng.normal(0.0, 0.03, n))
        v = np.cumsum(rng.normal(0.0, 0.03, n))
        f2 = float(rng.uniform(0.0, 2e-8))
        dt = float(rng.choice([300.0, 1800.0, 7200.0]))
        kappa0 = float(rng.choice([1e-7, 1e-6, 1e-5]))
        for eos in KAPPA_SHEAR_EOS_FORMS:
            base.append(dict(name=f"{eos} random {i}", eos=eos, dz=dz, u=u, v=v, T=T,
                             S=S * SA_SP if eos == "teos10" else S, f2=f2, dt=dt, kappa0=kappa0, rho0=rho0, g=g))
    return base


def write_kappa_shear_eos_cases(path: Path, cases: list[dict], params=None) -> None:
    """Write the cases for ``kappa_shear_driver eos`` (surface first); ``params`` = JacksonParams."""
    if params is None:
        from physics.mixing.kappa_shear import JacksonParams
        params = JacksonParams()
    reals = " ".join(f"{getattr(params, f):.17e}" for f in KAPPA_SHEAR_PARAM_FIELDS)
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write(f"{len(c['dz'])} {c['dt']:.17e} {c['f2']:.17e} {c['kappa0']:.17e} "
                    f"{KAPPA_SHEAR_EOS_FORMS[c['eos']]} {c['rho0']:.17e} {c['g']:.17e}\n")
            f.write(reals + "\n")
            f.write(f"{params.max_rino_it} {params.max_ks_it}\n")
            for key in ("dz", "u", "v", "T", "S"):
                f.write(" ".join(f"{x:.17e}" for x in np.asarray(c[key])[::-1]) + "\n")


# --------------------------------------------------------------------------------------------
# Drag-driven BBL mixing (MOM6 add_LOTW_BBL_diffusivity and set_BBL_TKE vs
# physics/mixing/bbl_mixing.py)
# --------------------------------------------------------------------------------------------
# Columns in the model orientation of the oracle (k = 0 at the bed); the writer flips them to
# MOM6's surface-first order and the reader flips the results back.  No thin-layer merge in any
# case (spec decision 6): the 1 mm layer goes to both codes as it is.

BBL_PARAM_FIELDS = ("bbl_effic", "max_decay", "von_karman", "kd_max")


def _bbl_stratified(nz: int, depth: float, n2_0: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """nz equal layers over ``depth`` m and N² = n2_0 exp(z/scale) at the interfaces (bed = 0)."""
    dz = np.full(nz, depth / nz)
    zi = np.concatenate(([0.0], np.cumsum(dz)))
    return dz, n2_0 * np.exp(zi / scale)


def bbl_cases() -> list[dict]:
    """
    Columns of the LOTW BBL comparison.  ``tke`` is TKE_col = BBL_EFFIC × energy (SD:1709);
    ``p`` is a dict of BBLParams overrides ({} = OM4 defaults).

    1. stratified, 40 × 5 m, N² = 1e-5 e^{z/100}, u* 1 cm/s, |f| 1e-4: the energy runs out
       about 15 m above the bed (energy-limited branch, SD:1765-1767);
    2. as 1 with N² = −2e-6 and 0 at the interfaces 5 and 10 m above the bed (the Ω² floor
       keeps W > 0);
    3. as 2 with LOTW_BBL_USE_OMEGA = False: both interfaces take KD_MAX (SD:1769-1772);
    4. f = 0: decay scale = BBL_MIXING_MAX_DECAY, Kd_wall = κ u* z (D − z)/D;
    5. a 1 mm layer at the bed and one 3 layers up (dz_int = 2.5005 m beside them, no merge);
    6. large u* (5 cm/s, |f| 1e-4: u*/|f| = 500 m > 200 m, so the 200 m decay cap) and
       TKE 1e-3 m³ s⁻³: Kd grows toward κ u*²/|f| until the energy is spent high in the column;
    7. ample energy, N² = 1e-8, 20 × 2 m: the energy never limits, Kd = Kd_wall at every
       interface (SD:1756-1757, uncut);
    8–15. seeded random columns (N = 5–60, dz 0.2–30 m, N² from −1e-6 to 1e-4,
       u* 0.1–5 cm/s, |f| 0–1.4e-4, TKE 1e-9–1e-4 m³ s⁻³; every other one without the Ω² floor).
    """
    cases = []
    dz, n2 = _bbl_stratified(40, 200.0, 1.0e-5, 100.0)
    cases.append(dict(name="stratified", dz=dz, n2=n2, ustar=0.01, absf=1.0e-4, tke=5.0e-6, p={}))
    n2c = n2.copy()
    n2c[1] = -2.0e-6
    n2c[2] = 0.0
    cases.append(dict(name="N2 <= 0 at two interfaces", dz=dz, n2=n2c, ustar=0.01, absf=1.0e-4,
                      tke=5.0e-6, p={}))
    cases.append(dict(name="N2 <= 0, no omega floor", dz=dz, n2=n2c, ustar=0.01, absf=1.0e-4,
                      tke=5.0e-6, p={"use_omega": False}))
    cases.append(dict(name="f = 0", dz=dz, n2=n2, ustar=0.01, absf=0.0, tke=5.0e-6, p={}))
    dzt = dz.copy()
    dzt[0] = 1.0e-3
    dzt[3] = 1.0e-3
    cases.append(dict(name="1 mm layers", dz=dzt, n2=n2, ustar=0.01, absf=1.0e-4, tke=5.0e-6, p={}))
    dz6, n26 = _bbl_stratified(50, 1000.0, 2.0e-6, 400.0)
    cases.append(dict(name="large u* (200 m cap)", dz=dz6, n2=n26, ustar=0.05, absf=1.0e-4,
                      tke=1.0e-3, p={}))
    dz7, n27 = _bbl_stratified(20, 40.0, 1.0e-8, 1.0e9)
    cases.append(dict(name="ample energy (uncut)", dz=dz7, n2=n27, ustar=0.02, absf=1.0e-4,
                      tke=1.0e-3, p={}))
    rng = np.random.default_rng(2006)
    for i in range(8):
        n = int(rng.integers(5, 61))
        cases.append(dict(name=f"random {i}", dz=rng.uniform(0.2, 30.0, n),
                          n2=rng.uniform(-1.0e-6, 1.0e-4, n + 1), ustar=float(rng.uniform(1e-3, 5e-2)),
                          absf=float(rng.choice([0.0, rng.uniform(0.0, 1.4e-4)])),
                          tke=float(10.0 ** rng.uniform(-9.0, -4.0)),
                          p={} if i % 2 == 0 else {"use_omega": False}))
    return cases


def write_bbl_cases(path: Path, cases: list[dict]) -> None:
    """
    Write the cases for ``bbl_driver lotw`` (surface first).  MOM6 receives
    visc%ustar_BBL = u* and visc%BBL_meanKE_loss = tke / BBL_EFFIC, so that its
    TKE_column = BBL_EFFIC × BBL_meanKE_loss (SD:1709) is pyOCEAN's tke.
    """
    from core.grid.horizontal import OMEGA
    from physics.mixing.bbl_mixing import BBLParams
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            p = BBLParams(**c["p"])
            f.write(f"{len(c['dz'])} {c['ustar']:.17e} {c['absf']:.17e} {c['tke'] / p.bbl_effic:.17e}\n")
            f.write(" ".join(f"{getattr(p, k):.17e}" for k in BBL_PARAM_FIELDS)
                    + f" {OMEGA:.17e} {int(p.use_omega)}\n")
            f.write(" ".join(f"{x:.17e}" for x in np.asarray(c["dz"])[::-1]) + "\n")
            f.write(" ".join(f"{x:.17e}" for x in np.asarray(c["n2"])[::-1]) + "\n")


def read_bbl_results(path: Path) -> list[dict]:
    """MOM6 Kd_BBL at the interfaces (k = 0 at the bed) of each case."""
    lines = [ln for ln in Path(path).read_text().split("\n") if ln.strip()]
    n = int(lines[0])
    return [dict(kd=np.array(lines[1 + i].split(), float)[::-1]) for i in range(n)]


def bbl_tke_cases() -> list[dict]:
    """
    Uniform flow in x (u = U in every layer, v = 0) for ``bbl_driver tke`` (MOM6 set_BBL_TKE):
    C_d, U [m/s], DRAG_BG_VEL U_bg [m/s], BBL thickness [m], layer thicknesses [m]
    (surface first), KV_BBL_MIN [m² s⁻¹].
    """
    dz = np.full(20, 10.0)
    return [dict(name="U 0.2, Ubg 0", cd=3.0e-3, u=0.2, ubg=0.0, bbl_thick=10.0, dz=dz, kv_min=0.0),
            dict(name="U 0.5, Ubg 0, 25 m BBL", cd=2.0e-3, u=0.5, ubg=0.0, bbl_thick=25.0, dz=dz, kv_min=0.0),
            dict(name="U 0.05, Ubg 0.1 (OM4)", cd=3.0e-3, u=0.05, ubg=0.1, bbl_thick=7.0, dz=dz, kv_min=0.0),
            # C_d |U| h = 4e-5 < KV_BBL_MIN = 1e-4 (DOME): kv_bbl clipped with bbl_thick unchanged
            # (CORRECT_BBL_BOUNDS = False, the default, SV:1103, 1142)
            dict(name="U 0.002, clipped by KV_BBL_MIN", cd=2.0e-3, u=0.002, ubg=0.0, bbl_thick=10.0, dz=dz,
                 kv_min=1.0e-4)]


def write_bbl_tke_cases(path: Path, cases: list[dict]) -> None:
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write(f"{len(c['dz'])} {c['cd']:.17e} {c['u']:.17e} {c['ubg']:.17e} "
                    f"{c['bbl_thick']:.17e} {c['kv_min']:.17e}\n")
            f.write(" ".join(f"{x:.17e}" for x in np.asarray(c["dz"])) + "\n")


def read_bbl_tke_results(path: Path) -> list[dict]:
    """visc%ustar_BBL and visc%BBL_meanKE_loss at the centre cell of each case."""
    lines = [ln for ln in Path(path).read_text().split("\n") if ln.strip()]
    n = int(lines[0])
    out = []
    for i in range(n):
        us, ke = (float(x) for x in lines[1 + i].split())
        out.append(dict(ustar_bbl=us, mean_ke_loss=ke))
    return out


# --------------------------------------------------------------------------- bottom N² of the BBL (find_N2)
BBL_N2_FORMS = {"linear": 1, "wright_full": 4, "teos10": 7}     # MOM_EOS.F90:161-168 (teos10 = ROQUET_RHO)
BBL_N2_LINEAR = dict(r0=1028.0, drdt=-0.2, drds=0.8)            # ρ = r0 + drdt T + drds S [kg m⁻³]


def bbl_n2_cases() -> list[dict]:
    """
    Columns for ``bbl_driver n2`` (MOM6 ``find_N2``, MOM_set_diffusivity.F90:1095-1270, with the
    bottom N²_bot overwrite of :1202-1258), bed first (k = 0 at the bed).  ρ₀ = 1035, g = 9.81.

    1. linear, 30 × 10 m, smooth stratification: N²_bot ≈ the local N² (small change).
    2. linear, 20 layers, the bed interface unstable (Δρ < 0, clipped to 0 in the bulk sum).
    3. linear, 12 random layers 2–60 m, random T: the bulk differs from both local values.
    4. wright_full, 40 layers 5–150 m to 3300 m, ocean-like θ, S_P.
    5. teos10, the same column in Θ, S_A.
    6. teos10, 3 layers: the smallest column with two interior interfaces.
    7. wright_full, 2 layers: one interior interface, MOM6's bulk equals the local value.
    8. linear, both bottom interfaces unstable: N²_bot = 0.
    """
    rng = np.random.default_rng(20261101)
    cases = []

    def add(name, eos, dz, T, S):
        cases.append(dict(name=name, eos=eos, dz=np.asarray(dz, float), T=np.asarray(T, float),
                          S=np.asarray(S, float), rho0=1035.0, g=9.81))

    z = np.arange(30) * 10.0
    add("linear smooth", "linear", np.full(30, 10.0), 4.0 + 16.0 * (1 - np.exp(-z / 120.0)), np.full(30, 35.0))
    T = np.linspace(2.0, 18.0, 20)
    T[0] = T[1] + 0.3
    add("linear unstable bed interface", "linear", np.full(20, 25.0), T, np.full(20, 35.0))
    add("linear random", "linear", rng.uniform(2.0, 60.0, 12), np.sort(rng.uniform(1.0, 20.0, 12)) +
        rng.normal(0.0, 0.4, 12), 35.0 + rng.normal(0.0, 0.05, 12))
    dz = np.geomspace(150.0, 5.0, 40)
    zc = np.cumsum(dz) - 0.5 * dz                                   # height above the bed
    T = 2.0 + 22.0 * np.exp(-(dz.sum() - zc) / 600.0)
    S = 34.9 + 1.2 * np.exp(-(dz.sum() - zc) / 400.0)
    add("wright_full deep", "wright_full", dz, T, S)
    add("teos10 deep", "teos10", dz, T, S * 35.16504 / 35.0)
    add("teos10 three layers", "teos10", [40.0, 25.0, 10.0], [3.0, 8.0, 15.0], [35.0, 35.3, 35.6])
    add("wright_full two layers", "wright_full", [80.0, 20.0], [4.0, 12.0], [34.9, 35.2])
    add("linear unstable bottom two", "linear", np.full(8, 20.0), [5.0, 4.8, 4.6, 6.0, 8.0, 10.0, 12.0, 14.0],
        np.full(8, 35.0))
    return cases


def bbl_n2_density(c: dict) -> np.ndarray:
    """Layer density of a linear case (``BBL_N2_LINEAR``), as MOM6's linear EOS."""
    lin = BBL_N2_LINEAR
    return lin["r0"] + lin["drdt"] * c["T"] + lin["drds"] * c["S"]


def write_bbl_n2_cases(path: Path, cases: list[dict]) -> None:
    """Write the cases for ``bbl_driver n2`` (surface first)."""
    lin = BBL_N2_LINEAR
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            f.write(f"{len(c['dz'])} {BBL_N2_FORMS[c['eos']]} {c['rho0']:.17e} {c['g']:.17e} "
                    f"{lin['r0']:.17e} {lin['drdt']:.17e} {lin['drds']:.17e}\n")
            for x in (c["dz"], c["T"], c["S"]):
                f.write(" ".join(f"{v:.17e}" for v in x[::-1]) + "\n")


def read_bbl_n2_results(path: Path) -> list[np.ndarray]:
    """MOM6 N2_int (nz+1,) [s⁻²] of each case, bed first."""
    lines = [ln for ln in Path(path).read_text().split("\n") if ln.strip()]
    n = int(lines[0])
    return [np.array(lines[1 + i].split(), float)[::-1] for i in range(n)]
