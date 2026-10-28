"""
N1 golden matrix (plan docs/superpowers/plans/2026-10-28-n1-component-core.md, task 0): every case of
``tests/n1_cases.py`` must reproduce, bit for bit, the state hashes produced by ``main`` before the N1
refactoring (``tests/data/n1_golden/hashes.json``, regenerated only with ``scripts/n1_make_golden.py``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.n1_cases import all_case_names, diff_fields, run_case

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "data" / "n1_golden" / "hashes.json").read_text())


@pytest.mark.parametrize("name", all_case_names())
def test_golden(name):
    got = run_case(name)
    for step, ref in GOLDEN[name].items():
        assert got[int(step)] == ref, f"{name} step {step}: fields differ {diff_fields(got[int(step)], ref)}"


def test_restart_from_main_file_continues_bit_for_bit():
    sys.path.insert(0, str(ROOT / "scripts"))
    from n1_make_golden import RESTART, restart_continue
    got = restart_continue(RESTART, write=False)
    assert got == GOLDEN["_restart_continue"], diff_fields(got, GOLDEN["_restart_continue"])


def test_restart_continuation_equals_continuous_run():
    # write at step 2 + read + 1 step = continuous step 3 (restart is bit for bit since stage A)
    assert GOLDEN["_restart_continue"] == GOLDEN["zstar_vi_arakawa"]["3"]
