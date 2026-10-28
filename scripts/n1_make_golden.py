#!/usr/bin/env python
"""
Generate (or check) the N1 golden matrix: state hashes of every case of ``tests/n1_cases.py`` after steps 1, 2, 3
and 10, the step-10 state of each case (for diagnosis only) and a reference restart written by the current code.

Run from ``main`` before the N1 refactoring (plan docs/superpowers/plans/2026-10-28-n1-component-core.md, task 0):

    python scripts/n1_make_golden.py           # write tests/data/n1_golden/ and tests/data/n1_restart_main.nc
    python scripts/n1_make_golden.py --check   # recompute and compare with the stored hashes (determinism)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.dynamics import sigma_baroclinic as sb  # noqa: E402
from core.restart import read_restart, write_restart  # noqa: E402
from tests.n1_cases import CASES, all_case_names, run_case, state_hashes  # noqa: E402

OUT = ROOT / "tests" / "data" / "n1_golden"
RESTART = ROOT / "tests" / "data" / "n1_restart_main.nc"
RESTART_CASE, RESTART_STEP = "zstar_vi_arakawa", 2


def restart_continue(path: Path, write: bool) -> dict[str, str]:
    """Write a restart at step 2 of RESTART_CASE (if ``write``), read it into a fresh model, step once."""
    sb.set_threads(1)
    m, st, wind = CASES[RESTART_CASE]()
    if write:
        for n in range(RESTART_STEP):
            st = m.step(st, *wind(n))
        write_restart(path, m, st)
    m2, _, wind2 = CASES[RESTART_CASE]()
    st2 = read_restart(path, m2)
    st2 = m2.step(st2, *wind2(RESTART_STEP))
    return state_hashes(st2, m2)


def save_states() -> None:
    sb.set_threads(1)
    for name in sorted(CASES):
        m, st, wind = CASES[name]()
        for n in range(10):
            st = m.step(st, *wind(n))
        arrays = {}
        for f in dataclasses.fields(st):
            v = getattr(st, f.name)
            if isinstance(v, np.ndarray):
                arrays[f.name] = v
        np.savez_compressed(OUT / f"{name}.npz", **arrays)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    golden = {name: {str(k): v for k, v in run_case(name).items()} for name in all_case_names()}
    if args.check:
        ref = json.loads((OUT / "hashes.json").read_text())
        same = [n for n in golden if golden[n] == ref[n]]
        same_rst = restart_continue(RESTART, write=False) == ref["_restart_continue"]
        print(f"{len(same)}/{len(golden)} identical; restart continuation {'identical' if same_rst else 'DIFFERS'}")
        for n in sorted(set(golden) - set(same)):
            print("  differs:", n)
        sys.exit(0 if len(same) == len(golden) and same_rst else 1)
    OUT.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    golden["_source"] = sha
    golden["_restart_continue"] = restart_continue(RESTART, write=True)
    (OUT / "hashes.json").write_text(json.dumps(golden, indent=1, sort_keys=True))
    save_states()
    print(f"wrote {len(golden) - 2} cases from {sha[:7]}")


if __name__ == "__main__":
    main()
