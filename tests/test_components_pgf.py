"""N1 task 3: EOS and pressure-gradient components (core/components/eos.py, pgf.py)."""
from __future__ import annotations

import numpy as np
import pytest

from core.components.eos import LinearEOS
from core.components.pgf import PressureGradient
from core.dynamics import sigma_baroclinic as sb
from core.dynamics.sigma_baroclinic import SigmaBaroclinic, reference_profile
from core.grid.horizontal import create_cartesian_grid
from core.model import threads


def _seamount(pgf, rho_ref=None, nz=8):
    dx, nx, ny = 8000.0, 16, 12
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=1e-4)
    x = (np.arange(nx) + 0.5) * dx - nx * dx / 2
    y = (np.arange(ny) + 0.5) * dx - ny * dx / 2
    H = 4500.0 - 4050.0 * np.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / 25e3**2)
    return SigmaBaroclinic(hg, H, dt=600.0, n_layers=nz, rho0=1000.0, t_coef=0.2, z0=None, pgf=pgf, rho_ref=rho_ref)


def test_linear_eos_is_the_model_density():
    eos = LinearEOS(rho0=1025.0, t_coef=0.2, t_ref=10.0)
    T = np.linspace(0.0, 30.0, 7)
    assert np.array_equal(eos.density(T), 1025.0 - 0.2 * (T - 10.0))


@pytest.mark.parametrize("scheme", ["jacobian", "twoterm", "fv"])
def test_pgf_component_is_what_the_model_uses(scheme):
    m = _seamount(scheme)
    z = m.layer_heights(np.zeros(m.H.shape))
    T = 20.0 + 0.005 * z + 0.3 * np.sin(z / 300.0)
    m.rho_ref = reference_profile(z, m.density(T))
    assert isinstance(m._pgf, PressureGradient) and m._pgf.rho_ref is m.rho_ref
    eta = 0.1 * np.ones(m.H.shape)
    a = m._pgf(m._geometry, eta, T, None)
    b = m._baroclinic_pgf(eta, T)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    assert np.abs(a[0]).max() > 0


def test_thread_list_is_shared():
    assert sb._THREADS is threads._THREADS
    sb.set_threads(1)
    assert threads._THREADS[0] == 1
