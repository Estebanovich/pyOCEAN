#!/usr/bin/env python
"""
Write (or check) the stratified σ + GLS hashes of ``tests/test_closure_component.py`` (N3 task 1): state and
closure hashes of two σ + GLS (gls_stratified) cases after 1, 10 and 50 steps.  Generated with the code at 7ee57ad,
before the closure was extracted from ``SigmaWetDry``; regenerated in N3 task 9b (log-law drag with the full
bed speed, class-(a) fix: both cases have z0 and v ≠ 0); regenerate only for an intended physics change.

    python scripts/n3_make_gls_hash.py           # write tests/data/n3_gls_sigma_hash.json
    python scripts/n3_make_gls_hash.py --check   # recompute and compare
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_closure_component import GLS_HASH_FILE, gls_sigma_hashes  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    got = gls_sigma_hashes()
    if a.check:
        ref = json.loads(GLS_HASH_FILE.read_text())
        print("identical" if got == ref else "DIFFERENT")
        sys.exit(0 if got == ref else 1)
    GLS_HASH_FILE.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GLS_HASH_FILE}")


if __name__ == "__main__":
    main()
