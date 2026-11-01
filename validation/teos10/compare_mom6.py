#!/usr/bin/env python
"""
pyOCEAN vs MOM6 on the TEOS-10 rest case in MOM6 mode (criterion 4, open item 2; ``mom6.py`` for the MOM6 side).

    python -m validation.teos10.compare_mom6 pyocean EOS COORD [N_LAYERS]   # one pyOCEAN run (≈ 30 s, 5 threads)
    python -m validation.teos10.compare_mom6 pyocean --all                  # the six runs, one after the other
    python -m validation.teos10.compare_mom6 table [--print]                # README table between the markers

The pyOCEAN runs are ``cases.rest_case("seamount", eos, coord, rho_ref=None)`` (the runs of ``run_pyocean.py``,
bit for bit: same case, same steps) re-run to record the metric of ``mom6.face_speed_max`` on both models: every
hour max |u|, |v| over all faces and layers, and over the faces whose two cells are both ≥ 1 m thick (a velocity in
a vanished layer moves no water; the first number is the criterion-4 metric of ``run_pyocean.py``).  The initial
|PGF| is the same pair of maxima of ``SigmaBaroclinic.pressure_gradient`` on the initial state; MOM6's is PFu, PFv
after one step with BEGW = 0 (``mom6.py`` docstring).  Outputs ``RUNS/mom6_rest/<eos>_<coord>/pyocean.npz``.

Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from core.dynamics.sigma_baroclinic import set_threads
from validation.teos10 import cases as C
from validation.teos10 import mom6

README = Path(__file__).parent / "README.md"
START, END = "<!-- mom6-rest:start -->", "<!-- mom6-rest:end -->"
COORDS = ("zstar", "hybrid")


def pyocean_path(eos: str, coord: str, n_layers: int | None = None, drag: bool = False) -> Path:
    return mom6.run_dir(eos, coord, "run", n_layers, drag).parent / "pyocean.npz"


def run_pyocean(eos: str, coord: str, n_layers: int | None = None, drag: bool = False) -> dict:
    """The pyOCEAN MOM6-mode rest run with the hourly metrics of ``mom6.face_speed_max``; saved and returned."""
    case = C.rest_case("seamount", eos, coord, rho_ref=None, n_layers=n_layers, drag=drag)
    m, st = case.model, case.state
    gx, gy = m.pressure_gradient(st.eta, st.T, st.h, st.S)
    pgf0, pgf0_thick = mom6.face_speed_max(gx, gy, st.h)
    every = int(round(3600.0 / m.dt))
    t, um, ut = [0.0], [0.0], [0.0]
    t0 = time.perf_counter()
    for n in range(1, case.n_steps + 1):
        st = m.step(st)
        if n % every == 0 or n == case.n_steps:
            if not np.isfinite(st.u).all():
                raise FloatingPointError(f"{case.name}: non-finite u at step {n}")
            a, b = mom6.face_speed_max(st.u, st.v, st.h)
            t.append(st.time)
            um.append(a)
            ut.append(b)
    out = dict(time=np.array(t), umax=np.array(um), umax_thick=np.array(ut), pgf0=pgf0, pgf0_thick=pgf0_thick,
               n_trunc=int(st.n_trunc), wall=time.perf_counter() - t0,
               argmax_u=np.array(np.unravel_index(np.abs(st.u).argmax(), st.u.shape)))
    p = pyocean_path(eos, coord, n_layers, drag)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **out)
    return out


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    with np.load(p) as d:
        return {k: d[k] for k in d.files}


def _at(r: dict, hours: float) -> float:
    i = int(np.argmin(np.abs(r["time"] - hours * 3600.0)))
    return float(r["umax"][i]), float(r["umax_thick"][i])


def table(n_layers: int | None = None, drag: bool = False) -> list[str]:
    """One row per (coordinate, EOS, model): initial |PGF| (all / thick faces), max|u| at 1 h, 1 d and the max over
    10 days (all / thick faces), truncations; the MOM6/pyOCEAN ratio of the 10-day max on thick faces."""
    out = ["| coord | EOS | model | \\|PGF\\| t=0 all / thick [m s⁻²] | max\\|u\\| 1 h all / thick | "
           "max\\|u\\| 1 d all / thick | max\\|u\\| 10 d all / thick [m s⁻¹] | truncations | MOM6/pyOCEAN thick 10 d |",
           "|---|---|---|---|---|---|---|---|---|"]
    for coord in COORDS:
        for eos in C.EOS_NAMES:
            p = _load(pyocean_path(eos, coord, n_layers, drag))
            q = mom6.load(eos, coord, n_layers, drag)
            for name, r in (("pyOCEAN", p), ("MOM6", q)):
                if r is None:
                    continue
                h1, d1 = _at(r, 1.0), _at(r, 24.0)
                ratio = ""
                if name == "MOM6" and p is not None:
                    ratio = f"{float(q['umax_thick'].max()) / float(p['umax_thick'].max()):.1f}"
                out.append(f"| {coord} | {eos} | {name} | {float(r['pgf0']):.2e} / {float(r['pgf0_thick']):.2e} | "
                           f"{h1[0]:.1e} / {h1[1]:.1e} | {d1[0]:.1e} / {d1[1]:.1e} | "
                           f"{float(r['umax'].max()):.2e} / {float(r['umax_thick'].max()):.2e} | "
                           f"{int(r['n_trunc'])} | {ratio} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("pyocean", "table"))
    ap.add_argument("case", nargs="*", help="EOS COORD [N_LAYERS] (pyocean)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--drag", action="store_true", help="the bed-drag sensitivity (cases.rest_case drag=True)")
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()
    if a.what == "pyocean":
        set_threads(a.threads)
        specs = ([(e, c, None) for c in COORDS for e in C.EOS_NAMES] if a.all
                 else [(a.case[0], a.case[1], int(a.case[2]) if len(a.case) > 2 else None)])
        for eos, coord, nl in specs:
            r = run_pyocean(eos, coord, nl, a.drag)
            print(f"pyOCEAN {eos} {coord} {nl or ''}: pgf0 {r['pgf0']:.3e}, max|u| {r['umax'].max():.3e} "
                  f"(thick {r['umax_thick'].max():.3e}), {r['wall']:.0f} s", flush=True)
        return
    text = "\n".join(["### Configuración del caso (lecho sin fricción, sin viscosidad horizontal)", ""] + table()
                     + ["", "### Sensibilidad: arrastre cuadrático en el fondo en los dos modelos "
                        "(C_d 3e-3, HBBL 10 m)", ""] + table(drag=True)
                     + ["", "### Resolución: híbrido con 50 capas, con arrastre", ""] + table(50, drag=True))
    if a.print:
        print(text)
        return
    src = README.read_text()
    i, j = src.index(START) + len(START), src.index(END)
    README.write_text(src[:i] + "\n" + text + "\n" + src[j:])
    print(f"table written to {README}")


if __name__ == "__main__":
    main()
