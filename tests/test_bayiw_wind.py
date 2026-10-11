"""
Wind forcing of the BayIW runners against the MITgcm two-stage setup:
stage 1 (0–60 h, data.stage1) reads merid_local_bay with periodicExternalForcing
(period 1200 s, records centred at (n + ½)·period, cycle = 180 records = 60 h);
stage 2 (60–120 h, data/data.stage2) has no wind file, so τ = 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation" / "bayiw_iw"))
from run_bayiw import WIND_PERIOD, wind_at  # noqa: E402

NREC = 180                                                   # 60 h of 1200 s records


@pytest.fixture
def records():
    r = np.zeros((NREC, 2, 3))
    r[:8] = np.arange(1, 9)[:, None, None]                   # a pulse in the first 8 records, as in the file
    return r


def test_records_centred(records):
    """Record n is reached at (n + ½)·period; halfway between two centres is their mean."""
    assert wind_at(2.5 * WIND_PERIOD, records)[0, 0] == pytest.approx(3.0)
    assert wind_at(3.0 * WIND_PERIOD, records)[0, 0] == pytest.approx(3.5)


def test_no_wind_in_stage2(records):
    """After the 60 h of stage 1 there is no wind: the pulse must not repeat at 60–62.5 h."""
    for t_h in (60.0, 60.2, 61.0, 62.5, 90.0, 119.9):
        assert not wind_at(t_h * 3600.0, records).any(), t_h


def test_cyclic_start(records):
    """Before the first centre MITgcm interpolates from the last record (cyclic within stage 1)."""
    records[-1] = 10.0
    assert wind_at(0.0, records)[0, 0] == pytest.approx(0.5 * (10.0 + 1.0))
