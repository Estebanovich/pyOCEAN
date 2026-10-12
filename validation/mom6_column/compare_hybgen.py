"""
Quantitative comparison of the pyOCEAN hybrid rule (regrid.hycom1_column, the HYCOM1
coordinate of MOM6) with HYCOM's hybgen (MOM6 port, MOM_hybgen_regrid.F90) on typical
columns.  The two are different algorithms (spec §8): hybgen moves interfaces by
entraining from neighbour layers with cushion-function minimum thicknesses (Bleck 2002;
Bleck & Benjamin 1993), HYCOM1 inverts a P1M density reconstruction for the target
densities with a z* floor (Adcroft et al. 2019).  The differences tell whether porting
hybgen is worth it (spec §3, out of scope of Plan A).

Set-up shared by both rules:
- 50 layers; one nominal z profile dz is hybgen's dp0k (minimum layer thicknesses) and
  HYCOM1's z* floor; minimum thickness 1e-3 m; hybgen nsigma = 0, isotop = 0, thkbot = 0,
  qhybrlx = 1 (no time relaxation).  Two configurations:
    A  dz = MITgcm BayIW delR (Σ = 599.9 m), as written in the plan.  In hybgen every layer
       is then at its minimum thickness in a 600 m column, so no interface can move
       (MOM_hybgen_regrid.F90:775-790, cushn); HYCOM1 uses dz only as a floor on interface
       depths and still moves the deep interfaces.
    B  HYCOM-like: dz = delR in the top 15 layers (Σ = 30.9 m), 1e-3 m below — fixed
       z-levels near the surface, isopycnal layers underneath;
    C  as B, with the 15 upper targets lighter than any water (OM4 style), so that HYCOM1
       keeps z-levels there too;
- interface targets evenly spaced over the BayIW density range (regrid.target_densities);
  hybgen layer targets are their midpoints;
- input grid: z* for the column depth; densities sampled at the layer centres.

Two measures: one regrid step from that input, and the state after 10 cycles of
regrid + PPM_H4 remap of density (the same remap for both rules), close to each rule's
fixed point.

Run: python validation/mom6_column/compare_hybgen.py  (needs build_reference.sh and the
BayIW MITgcm input on /Volumes/ESTEBAN_MIT).  Writes hybgen_comparison.txt here.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core.vertical import regrid as rg  # noqa: E402
from core.vertical import remap as rm  # noqa: E402

DRIVER = Path.home() / "pyOCEAN_refs" / "build" / "column" / "hybgen_driver"
RUN = Path("/Volumes/ESTEBAN_MIT/MITgcm/verification/BayIW_Rectan")
RHO_NIL, T_ALPHA, T_REF = 999.8, 2e-4, 16.582          # BayIW linear EOS (validation/bayiw_iw)
H_MIN = 1e-3
N_CYCLES = 10


def bayiw_profile() -> tuple[np.ndarray, np.ndarray]:
    """delR (surface first) [m] and the horizontally uniform BayIW density at level centres [kg m⁻³]."""
    txt = (RUN / "run_expand/data").read_text()
    block = re.search(r"delR\s*=(.*?)delYFile", txt, re.S).group(1)
    delr = np.array([float(x) for x in re.findall(r"\d+\.\d*|\d+", block)])
    T = np.fromfile(RUN / "input/feb_temp_50zlev_560x352.bin", ">f8").reshape(50, 352, 560)[:, 0, 0]
    return delr, RHO_NIL - RHO_NIL * T_ALPHA * (T - T_REF)


def columns(delr: np.ndarray, rho_b: np.ndarray) -> list[dict]:
    """Test columns: depth H [m] and a density function of depth d > 0 [m]."""
    zc = np.cumsum(delr) - 0.5 * delr
    bay = lambda d: np.interp(d, zc, rho_b)
    light_bottom = lambda d: bay(d) - 0.5 * (d > 164.0 - 10.0)          # 0.5 kg m⁻³ lighter bottom 10 m
    return [
        dict(name="estratificada 600 m", H=599.9, rho=lambda d: 999.8 + 2.1 * (1 - np.exp(-d / 200.0))),
        dict(name="mezclada 600 m", H=599.9, rho=lambda d: np.full_like(d, 1000.9)),
        dict(name="somera 30 m", H=30.0, rho=bay, blend=True),
        dict(name="164 m, fondo ligero", H=164.0, rho=light_bottom),
        dict(name="BayIW frontera 600 m", H=599.9, rho=bay),
    ]


def hybgen(cases: list[dict], dz: np.ndarray, tgt_layer: np.ndarray) -> list[np.ndarray]:
    """New interface depths (surface first, positive down) from MOM6 hybgen."""
    d = Path(tempfile.mkdtemp())
    with open(d / "in.txt", "w") as f:
        f.write(f"{len(cases)}\n")
        for c in cases:
            n = c["h"].size
            f.write(f"{n} 0\n")
            for a in (dz, dz):
                f.write(" ".join(f"{v:.17e}" for v in a) + "\n")
            f.write(f"{H_MIN:.17e} 0.0 1.0 0.0 {c['H']:.17e} {H_MIN:.17e}\n")
            for a in (c["h"], c["r"], tgt_layer):
                f.write(" ".join(f"{v:.17e}" for v in a) + "\n")
    subprocess.run([str(DRIVER), str(d / "in.txt"), str(d / "out.txt")], check=True, capture_output=True)
    lines = (d / "out.txt").read_text().split("\n")
    return [np.array(lines[1 + i].split(), float) for i in range(len(cases))]


def hycom1(c: dict, dz: np.ndarray, tgt_iface: np.ndarray, blend: bool) -> np.ndarray:
    """New interface depths (surface first, positive down) from regrid.hycom1_column."""
    n = c["h"].size
    zw = np.empty(n + 1)
    rg.hycom1_column(c["H"], 0.0, c["h"][::-1].copy(), c["r"][::-1].copy(), tgt_iface, dz,
                     np.full(n, np.inf), H_MIN, zw)
    if blend:
        sig, zs, out = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
        rg.sigma_column(c["H"], 0.0, np.full(n, 1.0 / n), H_MIN, sig)
        rg.zstar_column(c["H"], 0.0, dz, H_MIN, zs)
        rg.window_shallow_sigma(zw, sig, zs, c["H"], c["H"], 50.0, 100.0, H_MIN, out)
        zw = out
    return -zw[::-1]


def remap_density(h_old: np.ndarray, r: np.ndarray, x_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h_new = np.diff(x_new)
    r_new = np.empty(h_new.size)
    rm.remap_column(h_old, r, h_new, rm.PPM_H4, r_new)
    return h_new, r_new


def stats(x_ref: np.ndarray, x: np.ndarray) -> tuple[float, float, int, float]:
    """max |Δz|, mean |Δz| over interior interfaces [m], index of the max, max / local hybgen layer h."""
    d = np.abs(x - x_ref)[1:-1]
    k = int(np.argmax(d)) + 1
    h_loc = max(np.diff(x_ref)[min(k, x_ref.size - 2)], np.diff(x_ref)[k - 1])
    return d.max(), d.mean(), k, d.max() / max(h_loc, H_MIN)


def compare(cols: list[dict], dz: np.ndarray, tgt_iface: np.ndarray, tgt_layer: np.ndarray,
            delr: np.ndarray) -> list[tuple]:
    """One step and N_CYCLES regrid + remap cycles for both rules; rows of statistics."""
    for c in cols:
        zw = np.empty(delr.size + 1)
        rg.zstar_column(c["H"], 0.0, delr, H_MIN, zw)                   # input grid: z* on delR
        x0 = -zw[::-1]
        c["h"] = np.diff(x0)
        c["r"] = c["rho"](0.5 * (x0[1:] + x0[:-1]))
    xh = hybgen(cols, dz, tgt_layer)
    step = [(xh[i], hycom1(c, dz, tgt_iface, c.get("blend", False))) for i, c in enumerate(cols)]
    sh = [dict(H=c["H"], h=c["h"].copy(), r=c["r"].copy()) for c in cols]
    sp = [dict(H=c["H"], h=c["h"].copy(), r=c["r"].copy()) for c in cols]
    for _ in range(N_CYCLES):
        xh_c = hybgen(sh, dz, tgt_layer)
        for i, c in enumerate(cols):
            sh[i]["h"], sh[i]["r"] = remap_density(sh[i]["h"], sh[i]["r"], xh_c[i])
            xp = hycom1(sp[i], dz, tgt_iface, c.get("blend", False))
            sp[i]["h"], sp[i]["r"] = remap_density(sp[i]["h"], sp[i]["r"], xp)
    rows = []
    for i, c in enumerate(cols):
        a, b = step[i]
        xa = np.r_[0.0, np.cumsum(sh[i]["h"])]
        xb = np.r_[0.0, np.cumsum(sp[i]["h"])]
        rows.append((c["name"], stats(a, b), stats(xa, xb), int((np.diff(xa) > 2 * H_MIN).sum()),
                     int((np.diff(xb) > 2 * H_MIN).sum())))
    return rows


def main() -> None:
    delr, rho_b = bayiw_profile()
    tgt_iface = rg.target_densities(rho_b, delr.size)                  # 51, surface first
    dz_b = np.where(np.arange(delr.size) < 15, delr, H_MIN)
    # C: as B, with the 15 upper layer targets lighter than any water (OM4 style), the rest
    # evenly spaced over the profile range
    tgt_c = np.r_[rho_b.min() - 0.01 * np.arange(15, 0, -1), np.linspace(rho_b.min(), rho_b.max(), 36)]
    configs = {"A (dz = delR)": (delr, tgt_iface),
               "B (z en 31 m superiores)": (dz_b, tgt_iface),
               "C (B + objetivos ligeros arriba)": (dz_b, tgt_c)}
    hdr = (f"{'columna':24s} | {'1 paso: max|Δz| m':>17s} {'medio':>7s} {'k':>3s} {'/h':>6s} | "
           f"{'10 ciclos: max|Δz|':>18s} {'medio':>7s} {'k':>3s} {'/h':>6s} | capas>2h_min hybgen/pyOCEAN")
    lines = []
    for label, (dz, ti) in configs.items():
        lines += [f"Configuración {label}", hdr, "-" * len(hdr)]
        for name, s1, s2, na, nb in compare(columns(delr, rho_b), dz, ti, 0.5 * (ti[1:] + ti[:-1]), delr):
            lines.append(f"{name:24s} | {s1[0]:17.3f} {s1[1]:7.3f} {s1[2]:3d} {s1[3]:6.2f} | "
                         f"{s2[0]:18.3f} {s2[1]:7.3f} {s2[2]:3d} {s2[3]:6.2f} | {na:3d} / {nb:3d}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (HERE / "hybgen_comparison.txt").write_text(text)


if __name__ == "__main__":
    main()
