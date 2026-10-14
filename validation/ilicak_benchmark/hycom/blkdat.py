"""
blkdat.input, patch.input and limits of the HYCOM 2.3 Ilicak overflow cost case (spec
2026-10-13-ilicak-four-model-benchmark-design §A3).  blkdat lines are ``value 'name6'`` in the reading order of
HYCOM-src/blkdat.F90 (no dx00/dp00 lines: dp0k branch, blkdat.F90:233-290; nsigma = 0 so no ds0k line; no Stokes
lines without -DSTOKES, blkdat.F90:2524); patch.input in the fixed formats of mod_xc_mp.h:2535, 2604-2649.
Cost only: 7-term σ₀ EOS, no explicit mixing (vertmx = 0, mlflag = 0), hybgen with kdm hybrid layers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from validation.ilicak_benchmark.hycom import inputs

_HEAD = [("iversn", 23), ("iexpt", 10)]
_AFTER_KDM = [("nhybrd", None), ("nsigma", 0)]
_BEFORE_SIGMA = [("dp00i", 1.0), ("isotop", 5.0), ("oneta0", 0.1), ("saln0", 35.0), ("locsig", 0), ("kapref", 0),
                 ("thflag", 0), ("thbase", 25.0), ("vsigma", 0)]
_OUTPUT = [("iniflg", 2), ("jerlv0", 1), ("yrflag", 0), ("sshflg", 0), ("dsurfq", 0), ("diagfq", 0), ("proffq", 0),
           ("tilefq", 0), ("meanfq", 0), ("rstrfq", 0), ("bnstfq", 0), ("nestfq", 0), ("cplifq", 0)]
_REST = [
    ("incflg", 0), ("incstp", 1), ("incupf", 0), ("stfflg", 0), ("stfrdt", 0), ("stfrds", 0), ("stfrdv", 0),
    ("ra2fac", 0.125), ("wbaro", 0.125), ("btrlfr", 1), ("btrmas", 0), ("hybraf", 0), ("hybrlx", 1.0),
    ("hybiso", 0.01), ("hybthn", 1.0), ("hybthk", 0.0), ("hybmap", 3), ("hybflg", 0), ("advflg", 0), ("advtyp", 2),
    ("momtyp", 2), ("slip", 1.0), ("visco2", 0.0), ("visco4", 0.0), ("facdf4", 0.0), ("veldf2", None),
    ("veldf4", 0.0), ("thkdf2", 0.0), ("thkdf4", 0.0), ("temdf2", 0.0), ("temdfc", 1.0), ("vertmx", 0.0),
    ("cbar", 0.05), ("cb", 0.0), ("cbtidc", 0.0), ("drglim", 0.0), ("thkbot", 10.0), ("sigjmp", 0.02),
    ("tmljmp", 0.2), ("sefold", 30.0), ("thkmls", 15.0), ("thkmlt", 0.0), ("thkriv", 0.0), ("thkcdw", 3.0),
    ("thkfrz", 20.0), ("iceflg", 0), ("tfrz_0", 0.0), ("tfrz_s", -0.054), ("frzifq", -1.0), ("ticegr", 0.0),
    ("hicemn", 0.5), ("hicemx", 10.0), ("ishelf", 0), ("ntracr", 0), ("trcflg", 0), ("mtracr", 0), ("mstrcr", 0),
    ("tsofrq", 1), ("tofset", 0.0), ("sofset", 0.0), ("mlflag", 0), ("pensol", 0), ("dtrate", 999.0),
    ("thkmin", 0.0), ("dypflg", 0), ("mixfrq", 1), ("diapyc", 0.0), ("rigr", 0.25), ("ribc", 0.65),
    ("rinfty", 0.7), ("ricr", 0.45), ("bldmin", 0.0), ("bldmax", 1200.0), ("cekman", 0.7), ("cmonob", 1.0),
    ("bblkpp", 0), ("shinst", 0), ("dbdiff", 0), ("nonloc", 0), ("botdiw", 0), ("difout", 0), ("difsmo", 0),
    ("difm0", 5e-3), ("difs0", 5e-3), ("difmiw", 1e-4), ("difsiw", 1e-5), ("dsfmax", 1e-3), ("rrho0", 1.9),
    ("cs", 98.96), ("cstar", 10.0), ("cv", 0.0), ("c11", 5.0), ("hblflg", 2), ("niter", 2), ("langmr", 0),
    ("fltflg", 0), ("nfladv", 0), ("nflsam", 0), ("intpfl", 0), ("iturbv", 0), ("ismpfl", 0), ("tbvar", 0.0),
    ("tdecri", 0.0), ("lbflag", 0), ("lbmont", 0), ("tidflg", 0), ("tidein", 0), ("tidcon", 0), ("tidsal", 0.0),
    ("tiddrg", 0), ("thkdrg", 0.0), ("drgscl", 0.0), ("tidfm2", 0.0), ("tidfs2", 0.0), ("tidfk1", 0.0),
    ("tidfo1", 0.0), ("drgscm", 0.0), ("drgscs", 0.0), ("drgsck", 0.0), ("drgsco", 0.0), ("tidgen", 0),
    ("tidrmp", 0.0), ("tid_t0", 0.0), ("tidnud", 0), ("nudrmp", 0.0), ("nud_t0", 0.0), ("clmflg", 12),
    ("wndflg", 0), ("ocnscl", 0.0), ("ustflg", 1), ("flxflg", 0), ("empflg", 0), ("emptgt", 0.0), ("empbal", 0),
    ("dswflg", 0), ("albflg", 0), ("sssflg", 0), ("sssbal", 0), ("lwflag", 0), ("sstflg", 0), ("icmflg", 0),
    ("prsbas", 0.0), ("mslprf", 0), ("stroff", 0), ("flxoff", 0), ("flxsmo", 0), ("relax", 0), ("trcrlx", 0),
    ("priver", 0), ("epmass", 0)]


def _line(name: str, value: float | int) -> str:
    v = str(value) if isinstance(value, (int, np.integer)) else f"{float(value):.6g}"
    return f"  {v:<12} '{name:<6}' = "


def blkdat(kdm: int, baclin: float, batrop: float, nu_h: float, sigma: np.ndarray, dp0k: np.ndarray,
           idm: int = 202, jdm: int = 5) -> str:
    """blkdat.input text.  baclin, batrop: baroclinic and barotropic time steps [s] (baclin/batrop must be an even
    integer and baclin divide a day, as HYCOM's split-explicit barotropic loop and its day counter need); nu_h:
    Laplacian momentum viscosity [m² s⁻¹] → veldf2 = ν_h / Δx [m s⁻¹] (Δx = 1000 m); sigma: kdm layer target
    σ₀ [kg m⁻³ − 1000]; dp0k: kdm minimum z-layer thicknesses [m]."""
    ratio = baclin / batrop
    if abs(ratio - round(ratio)) > 1e-9 or round(ratio) % 2 or 86400.0 % baclin:
        raise ValueError(f"baclin/batrop = {ratio:g} must be an even integer and baclin must divide 86400 s")
    lines = ["Ilicak (2012) overflow, Petersen (2015) shelf: pyOCEAN four-model benchmark, cost only",
             "closed 202 x 5 (200 x 3 ocean), dx = dy = 1 km, 7-term sigma0, S = 35",
             "initial state from relax files (iniflg = 2, relax = 0)", "inputs: validation/ilicak_benchmark/hycom"]
    lines += [_line(n, v) for n, v in _HEAD + [("idm", idm), ("jdm", jdm), ("itest", 0), ("jtest", 0), ("kdm", kdm)]]
    lines += [_line(n, kdm if v is None else v) for n, v in _AFTER_KDM]
    lines += [_line("dp0k", float(d)) for d in dp0k]
    lines += [_line(n, v) for n, v in _BEFORE_SIGMA]
    lines += [_line("sigma", float(s)) for s in sigma]
    lines += [_line(n, v) for n, v in _OUTPUT + [("baclin", float(baclin)), ("batrop", float(batrop))]]
    lines += [_line(n, nu_h / 1000.0 if v is None else v) for n, v in _REST]
    return "\n".join(lines) + "\n"


def patch(npes: int, idm: int, jdm: int) -> str:
    """patch.input: npes tiles along i (one tile row), the larger tiles first (mod_xc_mp.h:2535 '(/8i6/)', then
    '(12x,8i6)' rows; nreg = 0, closed)."""
    sizes = [idm // npes + (1 if m < idm % npes else 0) for m in range(npes)]
    starts = list(np.cumsum([1] + sizes[:-1]))
    row = lambda label, vals: f"{label:<12}" + "".join(f"{v:6d}" for v in vals)        # noqa: E731
    return "\n".join([
        "  npes   npe   mpe   idm   jdm  ibig  jbig  nreg",
        "".join(f"{v:6d}" for v in (npes, npes, 1, idm, jdm, max(sizes), jdm, 0)),
        "",
        row("ispt(  1) =", starts),
        row("iipe(  1) =", sizes),
        "",
        row("jspt(  1) =", [1]),
        row("jjpe(  1) =", [jdm]),
    ]) + "\n"


def limits(hours: float) -> str:
    """limits: start and end model day."""
    return f"0.0 {hours / 24:.10f}\n"


def write_case(out: Path, npes: int, baclin: float, hours: float, nu_h: float = 100.0, kdm: int = 100,
               batrop: float = 1.0) -> Path:
    """Every input of the case in ``out``: grid, depth, relax files, blkdat.input, patch.input, limits."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs.write_grid(out)
    inputs.write_depth(out)
    inputs.write_relax(out, kdm)
    g = inputs.geometry()
    sigma = np.linspace(float(inputs.sig0(20.0, inputs.S0)), float(inputs.sig0(10.0, inputs.S0)), kdm)
    (out / "blkdat.input").write_text(blkdat(kdm, baclin, batrop, nu_h, sigma, np.asarray(g["dz"][:kdm], float),
                                             g["idm"], g["jdm"]))
    (out / "patch.input").write_text(patch(npes, g["idm"], g["jdm"]))
    (out / "limits").write_text(limits(hours))
    return out
