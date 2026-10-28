"""N1 task 13: restart organised by component sections; restart during the AB start-up."""
from __future__ import annotations

import warnings

import netCDF4

from core.restart import read_restart, write_restart
from tests.n1_cases import CASES, state_hashes


def test_restart_has_component_sections(tmp_path):
    warnings.simplefilter("ignore")
    for case in ("sigma_gls", "zstar_obc_radiating"):
        m, st, wind = CASES[case]()
        for n in range(2):
            st = m.step(st, *wind(n))
        path = tmp_path / f"{case}.nc"
        write_restart(path, m, st)
        with netCDF4.Dataset(path) as ds:
            comps = {name: ds[name].getncattr("component") for name in ds.variables}
        assert comps["eta"] == "state" and comps["ab_hist_u"] == "time_stepper"
        if case == "sigma_gls":
            assert comps["gls_tke"] == "closure"
        else:
            assert comps["T_ext"] == "open_boundary" and comps["fs_lu_data"] == "free_surface"


def test_restart_mid_ab_startup(tmp_path):
    # written after step 1 (AB1 history only) and continued: the AB2/AB3 start-up continues as in one run
    warnings.simplefilter("ignore")
    m, st, wind = CASES["zstar_vi_arakawa"]()
    a = st
    for n in range(3):
        a = m.step(a, *wind(n))
    m2, st2, _ = CASES["zstar_vi_arakawa"]()
    st2 = m2.step(st2, *wind(0))
    write_restart(tmp_path / "r.nc", m2, st2)
    m3, _, _ = CASES["zstar_vi_arakawa"]()
    b = read_restart(tmp_path / "r.nc", m3)
    for n in (1, 2):
        b = m3.step(b, *wind(n))
    assert state_hashes(a) == state_hashes(b)
