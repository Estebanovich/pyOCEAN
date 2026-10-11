"""
Tests for diagnostics/postmortem.py — locate and report a blow-up, and dump the
state, instead of a bare NaN error (forum request: "NaN checker that reports
where", Oceananigans #4333; HYCOM "neg dp" crashes).
"""

from __future__ import annotations

import numpy as np
import pytest

from diagnostics.postmortem import BlowupError, BlowupGuard


def _fields(ny=6, nx=8, nz=3):
    return dict(eta=np.zeros((ny, nx)), u=np.zeros((nz, ny, nx + 1)), v=np.zeros((nz, ny + 1, nx)))


def test_quiet_state_passes():
    guard = BlowupGuard(limits={"eta": 10.0, "u": 5.0})
    guard.check(step=3, time=360.0, **_fields())                      # no exception


def test_nan_is_located_and_the_state_dumped(tmp_path):
    f = _fields()
    f["u"][1, 4, 6] = np.nan
    lat = np.linspace(20, 21, 6)[:, None] * np.ones((1, 8))
    lon = np.linspace(-110, -109, 8)[None, :] * np.ones((6, 1))
    guard = BlowupGuard(limits={"u": 5.0}, dump_path=tmp_path / "blowup.npz", lat=lat, lon=lon,
                        depth=np.full((6, 8), 12.0), dt=60.0, dx=1000.0)
    with pytest.raises(BlowupError) as e:
        guard.check(step=17, time=1020.0, **f)
    msg = str(e.value)
    assert "step 17" in msg and "u" in msg and "NaN" in msg
    assert "k=1, j=4, i=6" in msg and "depth 12.0 m" in msg and "20.8" in msg      # lat of row 4
    d = np.load(tmp_path / "blowup.npz")
    assert np.isnan(d["u"][1, 4, 6]) and int(d["step"]) == 17


def test_limit_exceedance_reports_the_worst_cell_and_local_cfl():
    f = _fields()
    f["u"][0, 2, 3] = 9.0
    f["u"][2, 1, 1] = -7.0
    guard = BlowupGuard(limits={"u": 5.0}, dt=60.0, dx=500.0)
    with pytest.raises(BlowupError) as e:
        guard.check(step=1, time=60.0, **f)
    msg = str(e.value)
    assert "k=0, j=2, i=3" in msg and "9" in msg and "CFL 1.08" in msg              # 9 · 60 / 500


def test_history_of_maxima_is_kept_for_the_report():
    guard = BlowupGuard(limits={"eta": 1.0})
    for n in range(5):
        guard.check(step=n, time=60.0 * n, eta=np.full((2, 2), 0.1 * n))
    f = np.full((2, 2), 3.0)
    with pytest.raises(BlowupError) as e:
        guard.check(step=5, time=300.0, eta=f)
    assert "recent max|eta|" in str(e.value)
