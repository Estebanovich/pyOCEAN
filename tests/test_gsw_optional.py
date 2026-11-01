"""
TEOS-10 in the 3-D core, final review I1: ``gsw`` is an optional dependency.  The 3-D core imports and runs a
linear-EOS model without it (``physics/eos/convert.py`` imports ``gsw`` only inside the functions that use it,
as ``physics/eos/__init__.py`` does), and ``eos="teos10"``, whose conversions and freezing point need ``gsw``,
fails at construction with a clear message.  ``gsw`` is blocked in a fresh interpreter by
``sys.modules['gsw'] = None`` (then ``import gsw`` raises ImportError).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import sys, warnings
sys.modules["gsw"] = None                                   # gsw not installed
import numpy as np
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.components.eos import make_eos
from core.model.contract import PyOceanConfigWarning
from diagnostics.output import OutputWriter                # noqa: F401  (imports the conversions too)
NX, NY, NZ = 6, 5, 4
H = np.full((NY, NX), 50.0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", PyOceanConfigWarning)
    m = SigmaBaroclinic(create_cartesian_grid(nx=NX, ny=NY, dx=2000.0, dy=2000.0, f0=1e-4), H, dt=60.0,
                        n_layers=NZ, z0=None, closure="constant", nu_const=1e-4, ah=5.0, theta=0.6, d_min=0.05,
                        salinity=True, s_coef=0.78, s_ref=35.0, t_ref=10.0, vertical_coordinate="ale", pgf="fv",
                        regrid="zstar", regrid_params=dict(dz_top=np.full(NZ, 50.0 / NZ)))
T = np.full((NZ, NY, NX), -2.5)                             # below the linear Tf (−1.89 °C): not counted
st = m.initial_state(np.zeros((NY, NX)), T, S=np.full((NZ, NY, NX), 35.0))
for _ in range(3):
    st = m.step(st)
assert np.isfinite(st.T).all() and m.n_below_freezing == 0
try:
    make_eos("teos10")
except ImportError as e:
    assert "gsw" in str(e), e
else:
    raise AssertionError("make_eos('teos10') without gsw must raise ImportError")
assert sys.modules["gsw"] is None
print("OK")
"""


def test_linear_core_runs_without_gsw():
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith("OK"), out.stderr[-3000:]
