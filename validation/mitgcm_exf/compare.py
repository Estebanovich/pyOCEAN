"""
Point-by-point comparison of pyOCEAN ``forcing.bulk.bulk_ncar`` against
MITgcm ``pkg/exf`` (``exf_bulkformulae.F`` with ALLOW_BULK_LARGEYEAGER04,
optionally ALLOW_DRAG_LARGEYEAGER09).

Inputs : ``input/cases.npz`` (written by make_inputs.py, identical to the
         binary files MITgcm read).
MITgcm : ``run/<variant>/exfDiag.0000000001`` (diagnostics averaged over the
         single model step = exf fields computed at myIter = 0 from the
         initial θ(k=1) = SST).

Sign conventions are mapped to MITgcm's:
    EXFhs, EXFhl  > 0 into the ocean          (pyOCEAN q_sens, q_lat: same)
    EXFtaux/tauy  > 0 accelerates the ocean   (pyOCEAN tau_x, tau_y: same)
    EXFevap       > 0 evaporation [m s⁻¹]     (pyOCEAN evap: same)
    EXFlwnet      > 0 upward                  (= −pyOCEAN q_lw)
    EXFswnet      > 0 upward (negative heats) (= −pyOCEAN q_sw)
    EXFqnet       > 0 upward                  (= −(q_sw + q_lw + q_sens + q_lat))

The second half of the script contains a scratch NumPy port of
``exf_bulkformulae.F`` (``mit_ly04``) and of pyOCEAN's algorithm with
switchable choices (``py_ly04``).  They are used to attribute each systematic
difference to one known algorithmic/constant difference, walking from exact
MITgcm to exact pyOCEAN one change at a time.  pyOCEAN source is NOT modified.

Run:  /opt/anaconda3/bin/python compare.py [ly04|ly09|all]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MITGCM_PY = Path.home() / "pyOCEAN_refs/MITgcm/utils/python/MITgcmutils"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(MITGCM_PY))

from MITgcmutils import rdmds  # noqa: E402

from forcing.bulk import bulk_ncar, radiative_fluxes  # noqa: E402

# MITgcm pkg/exf defaults (exf_readparms.F:330-370, EXF_CONSTANTS.h:46,89)
MIT = dict(
    atmrho=1.2, atmcp=1005.0, flamb=2.5e6, cvapor_fac=640380.0, cvapor_exp=5107.4,
    humid_fac=0.606, gamma_blk=0.01, saltsat=0.98, cdrag_1=0.0027, cdrag_2=0.000142,
    cdrag_3=0.0000764, cdrag_8=-3.14807e-13, cdragMax=2.34e-3, umax=33.0,
    cstanton_1=0.0327, cstanton_2=0.0180, cdalton=0.0346, psim_fac=5.0, zref=10.0,
    hu=10.0, ht=2.0, umin=0.5, karman=0.4, gravity_mks=9.81, cen2kel=273.15,
    exf_albedo=0.1, ocean_emissivity=5.50e-8 / 5.670e-8, stefanBoltzmann=5.670e-8,
    niter_bulk=2, rhoConstFresh=1000.0,
)

FIELDS = ["EXFhs", "EXFhl", "EXFtaux", "EXFtauy", "EXFevap", "EXFqnet", "EXFlwnet", "EXFswnet",
          "EXFempmr", "EXFwspee", "EXFatemp", "EXFaqh", "EXFuwind", "EXFvwind", "EXFlwdn", "EXFswdn"]

# floor used in the normalised difference (py − mit) / max(|mit|, floor)
CD3_PY = 1e-3 / 13.09   # pyOCEAN cd_neutral_ly04 uses U/13.09 (LY04 eq 6a); MITgcm cdrag_3 = 7.64e-5

FLOOR = {"EXFhs": 1.0, "EXFhl": 1.0, "EXFtaux": 1e-3, "EXFtauy": 1e-3, "|tau|": 1e-3,
         "EXFevap": 1e-9, "EXFqnet": 1.0, "EXFlwnet": 1.0, "EXFswnet": 1.0, "EXFempmr": 1e-9}
COMPARED = ["EXFhs", "EXFhl", "EXFtaux", "EXFtauy", "|tau|", "EXFevap", "EXFqnet", "EXFlwnet",
            "EXFswnet", "EXFempmr"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_mitgcm(variant: str) -> dict[str, np.ndarray]:
    """Read the exf diagnostics of one run as a dict of (ny, nx) arrays."""
    arr = rdmds(str(HERE / "run" / variant / "exfDiag"), 1)
    out = {name: arr[k] for k, name in enumerate(FIELDS)}
    out["|tau|"] = np.hypot(out["EXFtaux"], out["EXFtauy"])
    return out


def pyocean_fluxes(c: dict[str, np.ndarray], high_wind: bool, cp_scale: float = 1.0
                   ) -> dict[str, np.ndarray]:
    """pyOCEAN fluxes with MITgcm-matching settings, in MITgcm sign conventions."""
    t = bulk_ncar(c["uwind"], c["vwind"], c["atemp"], c["aqh"], c["sst"],
                  rho_air=MIT["atmrho"], gamma=MIT["gamma_blk"], u_min=MIT["umin"],
                  z_u=MIT["hu"], z_t=MIT["ht"], n_iter=MIT["niter_bulk"], high_wind=high_wind)
    q_sw, q_lw = radiative_fluxes(c["swdown"], c["lwdown"], c["sst"],
                                  albedo=MIT["exf_albedo"], emissivity=MIT["ocean_emissivity"])
    hs = t.q_sens * cp_scale
    out = {
        "EXFhs": hs, "EXFhl": t.q_lat, "EXFtaux": t.tau_x, "EXFtauy": t.tau_y,
        "EXFevap": t.evap, "EXFlwnet": -q_lw, "EXFswnet": -q_sw,
        "EXFqnet": -(q_sw + q_lw + hs + t.q_lat), "EXFempmr": t.evap - c["precip"],
        "zeta": t.zeta,
    }
    out["|tau|"] = np.hypot(out["EXFtaux"], out["EXFtauy"])
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def ndiff(py: np.ndarray, mit: np.ndarray, field: str) -> np.ndarray:
    """Normalised difference (py − mit) / max(|mit|, floor)."""
    return (py - mit) / np.maximum(np.abs(mit), FLOOR[field])


def stats_line(d: np.ndarray) -> str:
    a = np.abs(d)
    return f"{np.median(d):+10.2e} {np.median(a):9.2e} {np.percentile(a, 95):9.2e} {a.max():9.2e}"


def print_table(py: dict, mit: dict, mask: np.ndarray, title: str) -> None:
    print(f"\n  {title}  (N = {mask.sum()})")
    print(f"  {'field':9s} | {'rel: median':>10s} {'med|.|':>9s} {'p95|.|':>9s} {'max|.|':>9s}"
          f" | {'abs: med|.|':>11s} {'p95|.|':>9s} {'max|.|':>9s}")
    for f in COMPARED:
        r = ndiff(py[f], mit[f], f)[mask]
        a = np.abs(py[f] - mit[f])[mask]
        print(f"  {f:9s} | {stats_line(r)} | {np.median(a):11.3e} {np.percentile(a, 95):9.3e} {a.max():9.3e}")


def bins(c: dict, zeta: np.ndarray) -> list[tuple[str, np.ndarray]]:
    ws = c["speed"]
    return [
        ("stable   ζ > +0.05", zeta > 0.05),
        ("neutral |ζ| ≤ 0.05", np.abs(zeta) <= 0.05),
        ("unstable ζ < −0.05", zeta < -0.05),
        ("U < 0.5 m/s (below umin)", ws < 0.5),
        ("0.5 ≤ U < 3", (ws >= 0.5) & (ws < 3)),
        ("3 ≤ U < 10", (ws >= 3) & (ws < 10)),
        ("10 ≤ U < 20", (ws >= 10) & (ws < 20)),
        ("20 ≤ U ≤ 35", ws >= 20),
    ]


# ---------------------------------------------------------------------------
# Scratch re-implementations for the diagnosis
# ---------------------------------------------------------------------------

def _cdn(u: np.ndarray, ly09: bool, p: dict) -> np.ndarray:
    """Neutral drag coefficient as coded in exf_bulkformulae.F (cdrag_* constants)."""
    cd = p["cdrag_1"] / u + p["cdrag_2"] + p["cdrag_3"] * u
    if ly09:
        cd = np.where(u - p["umax"] >= 0.0, p["cdragMax"], cd + p["cdrag_8"] * u ** 6)
    return cd


def mit_ly04(c: dict, *, ly09: bool = False, cp: float = 1005.0, humid: float = 0.606,
             clip_speed_in_flux: bool = False, cdrag_3: float | None = None,
             niter: int | None = None) -> dict:
    """
    NumPy port of MITgcm exf_bulkformulae.F, LY04 branch ("approach B": the
    transfer coefficients for heat and moisture are shifted to h_t, θ and q
    stay at h_t; ζ from atemp (absolute T, not θ) and aqh at h_t).

    Line references are to pkg/exf/exf_bulkformulae.F of the local clone.
    ``clip_speed_in_flux`` replaces the unclipped ``wspeed`` that MITgcm uses
    in U10N (l.428) and in the flux coefficient tau (l.457) by the clipped
    ``sh = max(wspeed, umin)`` (exf_wind.F:287), as LY04/pyOCEAN do.
    """
    p = dict(MIT)
    if cdrag_3 is not None:
        p["cdrag_3"] = cdrag_3
    if niter is not None:
        p["niter_bulk"] = niter
    tsf = c["sst"] + p["cen2kel"]
    ws = np.hypot(c["uwind"], c["vwind"])
    sh = np.maximum(ws, p["umin"])
    ws_flux = sh if clip_speed_in_flux else ws
    atemp, aqh = c["atemp"], c["aqh"]
    k = p["karman"]

    ssq = p["saltsat"] * p["cvapor_fac"] * np.exp(-p["cvapor_exp"] / tsf) / p["atmrho"]  # l.286-293
    deltap = atemp + p["gamma_blk"] * p["ht"] - tsf                                      # l.295
    delq = aqh - ssq
    stable = (deltap >= 0.0).astype(float)                                                # l.302 SIGN(.5,0)=+.5
    rdn = np.sqrt(_cdn(sh, ly09, p))                                                      # l.308-322
    ustar = rdn * sh
    rhn = (1 - stable) * p["cstanton_1"] + stable * p["cstanton_2"]                       # l.341
    tstar = rhn * deltap
    qstar = p["cdalton"] * delq
    zwln = np.log(p["hu"] / p["zref"])
    ztln = np.log(p["ht"] / p["zref"])
    czol = p["hu"] * k * p["gravity_mks"]
    tau = np.zeros_like(ws)
    rd = rdn

    for _ in range(p["niter_bulk"]):
        t0 = atemp * (1.0 + humid * aqh)                                                  # l.369
        huol = (tstar / t0 + qstar / (1.0 / humid + aqh)) * czol / ustar ** 2              # l.371
        huol = np.clip(huol, -10.0, 10.0)                                                 # l.375
        htol = huol * p["ht"] / p["hu"]
        stable = (huol >= 0.0).astype(float)
        xsq = np.sqrt(np.abs(1.0 - 16.0 * huol))
        x = np.sqrt(xsq)
        psimh = (-p["psim_fac"] * huol * stable
                 + (1 - stable) * (np.log((1 + 2 * x + xsq) * (1 + xsq) * 0.125)
                                   - 2 * np.arctan(x) + 0.5 * np.pi))                     # l.392
        xsq = np.sqrt(np.abs(1.0 - 16.0 * htol))
        psixh = -p["psim_fac"] * htol * stable + (1 - stable) * 2 * np.log(0.5 * (1 + xsq))  # l.408
        dz = (zwln - psimh) / k
        usn = ws_flux / (1.0 + rdn * dz)                                                  # l.428 (wspeed)
        usm = np.maximum(usn, p["umin"])
        rdn = np.sqrt(_cdn(usm, ly09, p))
        rd = rdn / (1.0 + rdn * dz)
        ustar = rd * sh                                                                    # l.457 (sh)
        tau = p["atmrho"] * rd * ws_flux                                                   # l.462 (wspeed)
        rhn = (1 - stable) * p["cstanton_1"] + stable * p["cstanton_2"]
        rh = rhn / (1.0 + rhn * (ztln - psixh) / k)                                        # l.473
        re = p["cdalton"] / (1.0 + p["cdalton"] * (ztln - psixh) / k)
        qstar = re * delq
        tstar = rh * deltap

    hs = cp * tau * tstar
    hl = p["flamb"] * tau * qstar
    evap = -tau * qstar / p["rhoConstFresh"]
    taux = tau * rd * c["uwind"]
    tauy = tau * rd * c["vwind"]
    return _pack(hs, hl, taux, tauy, evap, c, huol)


def py_ly04(c: dict, *, ly09: bool = False, cp: float = 1000.5, humid: float = 0.608,
            zeta_from: str = "zu", cdrag_3: float | None = None, niter: int | None = None) -> dict:
    """
    NumPy port of pyOCEAN bulk_ncar ("approach C": θ and q shifted from z_t
    to z_u, coefficients at z_u), with switchable constants.

    ``zeta_from='zu'``   : ζ from θ_v and q at z_u, updated each pass (pyOCEAN)
    ``zeta_from='atemp'``: ζ from absolute atemp and aqh at h_t, fixed (as MITgcm)
    """
    p = dict(MIT)
    if cdrag_3 is not None:
        p["cdrag_3"] = cdrag_3
    if niter is not None:
        p["niter_bulk"] = niter
    k, g, zu, zt = p["karman"], p["gravity_mks"], p["hu"], p["ht"]
    rho = p["atmrho"]
    sst_k = c["sst"] + p["cen2kel"]
    theta = c["atemp"] + p["gamma_blk"] * zt
    q = c["aqh"]
    du, dv = c["uwind"], c["vwind"]
    speed = np.maximum(np.hypot(du, dv), p["umin"])
    q_sfc = p["saltsat"] * p["cvapor_fac"] / rho * np.exp(-p["cvapor_exp"] / sst_k)
    log_u, log_tu = np.log(zu / 10.0), np.log(zt / zu)
    d_theta, d_q = theta - sst_k, q - q_sfc
    cdn = _cdn(speed, ly09, p)
    sq_cdn = np.sqrt(cdn)
    chn = np.where(d_theta > 0, 18.0e-3, 32.7e-3) * sq_cdn
    cen = 34.6e-3 * sq_cdn
    sq_cd = sq_cdn
    ustar, tstar, qstar = sq_cd * speed, chn / sq_cd * d_theta, cen / sq_cd * d_q
    theta_zu, q_zu = theta, q
    ch = ce = None
    for _ in range(p["niter_bulk"]):
        if zeta_from == "zu":
            tv, qz = theta_zu * (1 + humid * q_zu), q_zu
        else:
            tv, qz = c["atemp"] * (1 + humid * q), q
        zeta = k * g * zu / ustar ** 2 * (tstar / tv + qstar / (qz + 1.0 / humid))
        zeta = np.clip(zeta, -10, 10)
        x = (1 - 16 * np.minimum(zeta, 0)) ** 0.25
        psim = np.where(zeta >= 0, -5 * zeta, 2 * np.log((1 + x) / 2) + np.log((1 + x * x) / 2)
                        - 2 * np.arctan(x) + np.pi / 2)
        psih = np.where(zeta >= 0, -5 * zeta, 2 * np.log((1 + x * x) / 2))
        zt_ = zeta * zt / zu
        xt = (1 - 16 * np.minimum(zt_, 0)) ** 0.25
        psih_t = np.where(zt_ >= 0, -5 * zt_, 2 * np.log((1 + xt * xt) / 2))
        u10n = np.maximum(speed / (1 + sq_cdn / k * (log_u - psim)), p["umin"])
        shift = log_tu + psih - psih_t
        theta_zu = theta - tstar / k * shift
        q_zu = q - qstar / k * shift
        d_theta, d_q = theta_zu - sst_k, q_zu - q_sfc
        cdn = _cdn(u10n, ly09, p)
        sq_cdn = np.sqrt(cdn)
        chn = np.where(zeta > 0, 18.0e-3, 32.7e-3) * sq_cdn
        cen = 34.6e-3 * sq_cdn
        sq_cd = sq_cdn / (1 + sq_cdn / k * (log_u - psim))
        ch = chn * (sq_cd / sq_cdn) / (1 + chn / (k * sq_cdn) * (log_u - psih))
        ce = cen * (sq_cd / sq_cdn) / (1 + cen / (k * sq_cdn) * (log_u - psih))
        ustar, tstar, qstar = sq_cd * speed, ch / sq_cd * d_theta, ce / sq_cd * d_q
    cd = sq_cd ** 2
    hs = rho * cp * ch * speed * d_theta
    wf = rho * ce * speed * d_q
    return _pack(hs, p["flamb"] * wf, rho * cd * speed * du, rho * cd * speed * dv,
                 -wf / p["rhoConstFresh"], c, zeta)


def _pack(hs, hl, taux, tauy, evap, c, zeta) -> dict:
    """Turbulent fluxes + MITgcm radiation (exf_radiation.F with EXF_LWDOWN_WITH_EMISSIVITY)."""
    tsf = c["sst"] + MIT["cen2kel"]
    lw = MIT["ocean_emissivity"] * (MIT["stefanBoltzmann"] * tsf ** 4 - c["lwdown"])
    sw = -c["swdown"] * (1 - MIT["exf_albedo"])
    return {"EXFhs": hs, "EXFhl": hl, "EXFtaux": taux, "EXFtauy": tauy, "|tau|": np.hypot(taux, tauy),
            "EXFevap": evap, "EXFlwnet": lw, "EXFswnet": sw, "EXFqnet": -hs - hl + lw + sw,
            "EXFempmr": evap - c["precip"], "zeta": zeta}


def summary_row(a: dict, b: dict, fields=("EXFhs", "EXFhl", "|tau|", "EXFevap", "EXFqnet"),
                mask: np.ndarray | None = None) -> str:
    """max |normalised difference| of a vs b for the main fields."""
    cols = []
    for f in fields:
        d = np.abs(ndiff(a[f], b[f], f))
        if mask is not None:
            d = d[mask]
        cols.append(f"{np.median(d):8.1e}/{d.max():8.1e}")
    return "  ".join(cols)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_variant(variant: str, c: dict, figs: list) -> None:
    ly09 = variant == "ly09"
    mit = load_mitgcm(variant)
    py = pyocean_fluxes(c, high_wind=ly09)
    py_cp = pyocean_fluxes(c, high_wind=ly09, cp_scale=MIT["atmcp"] / 1000.5)
    bar = "=" * 100
    print(f"\n{bar}\n VARIANT {variant.upper()}  (ALLOW_BULK_LARGEYEAGER04"
          f"{' + ALLOW_DRAG_LARGEYEAGER09' if ly09 else ''})\n{bar}")

    # --- sanity: MITgcm saw exactly our inputs, and gcmSST = SST
    tsf_mit = ((mit["EXFlwnet"] / MIT["ocean_emissivity"] + mit["EXFlwdn"])
               / MIT["stefanBoltzmann"]) ** 0.25 - MIT["cen2kel"]
    print(" input echo max|Δ|: atemp {:.1e} K, aqh {:.1e}, wspeed {:.1e} m/s, uwind {:.1e}, "
          "SST recovered from EXFlwnet {:.1e} K".format(
              np.abs(mit["EXFatemp"] - c["atemp"]).max(), np.abs(mit["EXFaqh"] - c["aqh"]).max(),
              np.abs(mit["EXFwspee"] - c["speed"]).max(), np.abs(mit["EXFuwind"] - c["uwind"]).max(),
              np.abs(tsf_mit - c["sst"]).max()))

    # --- 1. port checks: scratch ports reproduce both codes to round-off
    port = mit_ly04(c, ly09=ly09)
    print("\n [check] scratch port of exf_bulkformulae.F vs MITgcm run, max|normalised Δ| per field:")
    print("   " + "  ".join(f"{f}={np.abs(ndiff(port[f], mit[f], f)).max():.1e}" for f in COMPARED))
    pyport = py_ly04(c, ly09=ly09, cdrag_3=CD3_PY)
    print(" [check] scratch port of pyOCEAN bulk_ncar vs pyOCEAN, max|normalised Δ| "
          "(LW/Qnet differ only by σ 5.670e-8 vs 5.670374e-8 in _pack):")
    print("   " + "  ".join(f"{f}={np.abs(ndiff(pyport[f], py[f], f)).max():.1e}" for f in COMPARED))

    # --- 2. headline tables
    print("\n [A] pyOCEAN (as is) vs MITgcm.  rel = (py − mit)/max(|mit|, floor); "
          "floors: 1 W/m², 1e-3 N/m², 1e-9 m/s")
    print_table(py, mit, np.ones_like(c["speed"], bool), "ALL cases")
    print("\n [B] pyOCEAN with Q_S rescaled by atmcp/CP_AIR = 1005/1000.5 (constant difference removed)")
    print_table(py_cp, mit, np.ones_like(c["speed"], bool), "ALL cases")
    for name, m in bins(c, py["zeta"]):
        print_table(py_cp, mit, m, name)

    # --- 3. worst cases (cp-rescaled, U ≥ umin so the calm-wind quirk is excluded)
    print("\n [C] worst cases (cp-rescaled pyOCEAN), U ≥ 0.5 m/s and U < 0.5 m/s separately")
    hdr = (f"   {'U':>6s} {'ΔT':>6s} {'RH':>5s} {'SST':>5s} {'ζ_mit':>7s} {'ζ_py':>7s} |"
           f" {'mit':>11s} {'py':>11s} {'rel':>9s}")
    for f in ("EXFhs", "EXFhl", "|tau|"):
        for label, m in (("U≥0.5", c["speed"] >= 0.5), ("U<0.5", c["speed"] < 0.5)):
            r = ndiff(py_cp[f], mit[f], f)
            idx = np.argsort(np.where(m, -np.abs(r), 0.0), axis=None)[:4]
            print(f"  {f} ({label}):\n{hdr}")
            for i in idx:
                j, k = np.unravel_index(i, r.shape)
                print(f"   {c['speed'][j, k]:6.2f} {c['dT'][j, k]:+6.2f} {100 * c['rh'][j, k]:5.1f}"
                      f" {c['sst'][j, k]:5.1f} {port['zeta'][j, k]:+7.3f} {py['zeta'][j, k]:+7.3f} |"
                      f" {mit[f][j, k]:+11.4e} {py_cp[f][j, k]:+11.4e} {r[j, k]:+9.2e}")

    # --- 4. attribution chain: MITgcm → pyOCEAN, one change at a time
    print("\n [D] attribution chain: median/max |normalised Δ| of each step vs the previous one")
    print(f"   {'step':58s} {'EXFhs':>17s}  {'EXFhl':>17s}  {'|tau|':>17s}  {'EXFevap':>17s}  {'EXFqnet':>17s}")
    kw = dict(ly09=ly09, cp=1000.5, humid=0.608, cdrag_3=CD3_PY)
    steps = [
        ("S0 MITgcm port (exact)", port),
        ("S1 + cp 1005 → 1000.5", mit_ly04(c, ly09=ly09, cp=1000.5)),
        ("S2 + humid_fac 0.606 → 0.608", mit_ly04(c, ly09=ly09, cp=1000.5, humid=0.608)),
        ("S3 + cdrag_3 7.64e-5 → 1e-3/13.09", mit_ly04(c, **kw)),
        ("S4 + clipped U in U10N & fluxes (LY04)", mit_ly04(c, **kw, clip_speed_in_flux=True)),
        ("S5 + approach C (shift θ,q to z_u), ζ from atemp@h_t",
         py_ly04(c, ly09=ly09, cdrag_3=CD3_PY, zeta_from="atemp")),
        ("S6 + ζ from θ_v, q at z_u (= pyOCEAN)", pyport),
    ]
    ok = c["speed"] >= 0.5
    prev = mit
    for name, s in steps:
        print(f"   {name:58s} {summary_row(s, prev)}")
        print(f"   {'   (U ≥ 0.5 only)':58s} {summary_row(s, prev, mask=ok)}")
        prev = s
    print(f"   {'S6 vs pyOCEAN bulk_ncar (round-off check)':58s} {summary_row(pyport, py)}")

    # --- 5. classify every remaining >1 % difference (U ≥ 0.5, cp-rescaled)
    #  B   = MITgcm algorithm with pyOCEAN constants and clipped U   (S4)
    #  Ca  = pyOCEAN algorithm with ζ defined as in MITgcm           (S5)
    #  C   = pyOCEAN                                                  (S6)
    #  X∞  = same, iterated n_conv times instead of niter_bulk = 2
    main = ("EXFhs", "EXFhl", "|tau|", "EXFevap")

    def dmax(a: dict, b: dict) -> np.ndarray:
        return np.max([np.abs(ndiff(a[f], b[f], f)) for f in main], axis=0)

    worst = dmax(py_cp, mit)
    flagged = ok & (worst > 0.01)
    n_conv = 60
    kwb = dict(kw, clip_speed_in_flux=True)
    b_inf, b_inf1 = mit_ly04(c, **kwb, niter=n_conv), mit_ly04(c, **kwb, niter=n_conv - 1)
    kwc = dict(ly09=ly09, cdrag_3=CD3_PY)
    ca_inf = py_ly04(c, **kwc, zeta_from="atemp", niter=n_conv)
    ca_inf1 = py_ly04(c, **kwc, zeta_from="atemp", niter=n_conv - 1)
    c_inf, c_inf1 = py_ly04(c, **kwc, niter=n_conv), py_ly04(c, **kwc, niter=n_conv - 1)
    converged = (dmax(b_inf, b_inf1) < 1e-3) & (dmax(ca_inf, ca_inf1) < 1e-3) & (dmax(c_inf, c_inf1) < 1e-3)
    same_fixed_point = dmax(ca_inf, b_inf) < 0.01      # approach B ≡ approach C at convergence
    same_zeta_def = dmax(c_inf, ca_inf) < 0.01         # ζ definition irrelevant at convergence
    print(f"\n [E] approach B vs C iterated to convergence ({n_conv} passes), same ζ definition, "
          f"converged cases with U ≥ 0.5 (N = {(ok & converged).sum()}):\n   "
          + summary_row(ca_inf, b_inf, mask=ok & converged))
    cats = {
        "n_iter=2 truncation (B ≡ C once converged, ζ definition irrelevant)":
            flagged & converged & same_fixed_point & same_zeta_def,
        "ζ definition (θ_v,q at z_u vs atemp,aqh at h_t) still matters at convergence":
            flagged & converged & same_fixed_point & ~same_zeta_def,
        "iteration has no fixed point (limit cycle across the ζ = 0 C_hN switch or ζ clip)":
            flagged & ~converged,
        "unexplained": flagged & converged & ~same_fixed_point,
    }
    zp = py["zeta"]
    print(f"\n [F] cases with U ≥ 0.5 and >1 % difference in any of {main} (cp-rescaled): "
          f"{flagged.sum()} / {ok.sum()}")
    for name, m in cats.items():
        rng = (f"  ζ_py ∈ [{zp[m].min():+.2f}, {zp[m].max():+.2f}], U ∈ [{c['speed'][m].min():.2f}, "
               f"{c['speed'][m].max():.2f}], max diff {worst[m].max():.2f}") if m.any() else ""
        print(f"   {m.sum():4d}  {name}{rng}")
    for j, k in zip(*np.nonzero(cats["unexplained"])):
        print(f"     unexplained: U={c['speed'][j, k]:.2f} ΔT={c['dT'][j, k]:+.2f} "
              f"RH={100 * c['rh'][j, k]:.0f} SST={c['sst'][j, k]:.1f} ζ={zp[j, k]:+.3f} worst={worst[j, k]:.2e}")
    m = cats["n_iter=2 truncation (B ≡ C once converged, ζ definition irrelevant)"]
    if m.any():
        b2 = mit_ly04(c, **kwb)
        print(f"   truncation cases: distance to the converged solution, median/max: "
              f"MITgcm(2 it) {np.median(dmax(b2, b_inf)[m]):.1e}/{dmax(b2, b_inf)[m].max():.1e}, "
              f"pyOCEAN(2 it) {np.median(dmax(pyport, c_inf)[m]):.1e}/{dmax(pyport, c_inf)[m].max():.1e}")
    print(f"   cases with no fixed point overall (U ≥ 0.5): {(ok & ~converged).sum()}, of which "
          f"{(ok & ~converged & (np.abs(zp) < 0.3)).sum()} near-neutral |ζ|<0.3")

    figs.append((variant, py_cp, mit))


def make_figure(figs: list, c: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = ["EXFhs", "EXFhl", "EXFevap", "EXFtaux", "EXFtauy", "EXFlwnet", "EXFswnet", "EXFqnet"]
    units = {"EXFhs": "W m⁻²", "EXFhl": "W m⁻²", "EXFevap": "m s⁻¹", "EXFtaux": "N m⁻²",
             "EXFtauy": "N m⁻²", "EXFlwnet": "W m⁻²", "EXFswnet": "W m⁻²", "EXFqnet": "W m⁻²"}
    variant, py, mit = figs[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2), constrained_layout=True)
    ws = c["speed"]
    for ax, f in zip(axes.flat, fields):
        sc = ax.scatter(mit[f].ravel(), py[f].ravel(), c=np.log10(ws).ravel(), s=7, cmap="viridis",
                        vmin=np.log10(0.2), vmax=np.log10(35))
        lo = min(mit[f].min(), py[f].min())
        hi = max(mit[f].max(), py[f].max())
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.8, zorder=0)
        r = np.abs(ndiff(py[f], mit[f], f))
        ax.set_title(f"{f} [{units[f]}]\nmedian |rel Δ| {np.median(r):.1e}, "
                     f"U≥0.5 max {r[ws >= 0.5].max():.1e}", fontsize=9)
        ax.set_xlabel("MITgcm exf", fontsize=8)
        ax.set_ylabel("pyOCEAN bulk_ncar", fontsize=8)
        ax.tick_params(labelsize=7)
    cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.01)
    ticks = [0.2, 0.5, 1, 3, 10, 35]
    cb.set_ticks(np.log10(ticks))
    cb.set_ticklabels([str(t) for t in ticks])
    cb.set_label("wind speed [m s⁻¹]")
    fig.suptitle(f"pyOCEAN bulk_ncar vs MITgcm pkg/exf ({variant.upper()}), 1000 cases; "
                 "Q_S rescaled by 1005/1000.5; signs in MITgcm convention", fontsize=11)
    fig.savefig(HERE / "compare.png", dpi=130)
    print(f"\nfigure written to {HERE / 'compare.png'}")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    variants = ["ly04", "ly09"] if arg == "all" else [arg]
    c = dict(np.load(HERE / "input" / "cases.npz"))
    figs: list = []
    for v in variants:
        if (HERE / "run" / v / "exfDiag.0000000001.001.001.data").exists() or \
           (HERE / "run" / v / "exfDiag.0000000001.data").exists():
            run_variant(v, c, figs)
        else:
            print(f"(skipping {v}: no MITgcm output; run ./build_and_run.sh {v})")
    if figs:
        make_figure(figs, c)


if __name__ == "__main__":
    main()
