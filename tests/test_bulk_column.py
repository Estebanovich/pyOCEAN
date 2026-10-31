"""
N2b task 7 (validation/bulk_column): the ERA5 column against MITgcm (LY04) and CROCO (COARE 3.0).

1. Case definition: the values of ``cases.py`` are the ones in the reference inputs (MITgcm namelists, CROCO
   croco.in and source patches), so the three models run the same case.
2. Input conversions (CROCO units, MITgcm record layout) and the mixed-layer metric, without files.
3. With the runs on disk (skipped otherwise): every model's output is read, the column stayed 1-D, the inputs are
   the ERA5 series at the point, and the comparison tables are finite.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from validation.bulk_column import cases as C
from validation.bulk_column import compare, forcing

HERE = Path(__file__).resolve().parents[1] / "validation" / "bulk_column"
MIT_IN = HERE / "mitgcm" / "input"


def _nml(path: Path) -> dict[str, str]:
    """name → raw value of a Fortran namelist file (one assignment per line, comments '#')."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0]
        for m in re.finditer(r"(\w+)\s*=\s*(.*?)\s*,?\s*(?=\b\w+\s*=|$)", line.strip()):
            out[m.group(1)] = m.group(2).strip().rstrip(",").strip().strip("'")
    return out


# ------------------------------------------------------------------------------------------------ case definition
def test_point_and_column():
    assert 40.0 < C.LAT < 40.25 and -73.25 < C.LON < -73.0          # bilinear ERA5 stencil = the 4 sea nodes
    assert C.N * 1.0 == C.DEPTH and C.DAYS >= 30 and C.DAYS_JULY >= 30
    assert C.F0 == pytest.approx(2 * 7.2921e-5 * np.sin(np.deg2rad(40.24)))
    T, S = C.initial_TS()
    assert T.shape == S.shape == (C.N,)
    assert T[-1] == pytest.approx(np.interp(0.5, C.WOA_DEPTH, C.WOA_T)) and T[0] < T[-1]   # bed first
    assert np.all(np.diff(C.rho_linear(T, S)) < 0)                   # stable: ρ decreases upward


def test_woa_constants_match_the_download():
    p = C.RUNS / "woa" / "woa23_decav_t07_04_subset.npz"
    if not p.exists():
        pytest.skip("WOA subset not on disk")
    for v, ref in (("t", C.WOA_T), ("s", C.WOA_S)):
        d = np.load(C.RUNS / "woa" / f"woa23_decav_{v}07_04_subset.npz")
        j = int(np.argmin(abs(d["lat"] - C.WOA_CELL[0])))
        i = int(np.argmin(abs(d["lon"] - C.WOA_CELL[1])))
        k = np.isin(d["depth"], C.WOA_DEPTH)
        np.testing.assert_allclose(d["an"][k, j, i], ref, atol=0.006)


def test_mitgcm_namelists_match_cases():
    d = _nml(MIT_IN / "data")
    assert d["eosType"] == "LINEAR"
    assert float(d["tAlpha"]) == C.ALPHA and float(d["sBeta"]) == C.BETA
    assert d["tRef"] == f"{C.N}*{C.T_REF}" and d["sRef"] == f"{C.N}*{C.S_REF}"
    assert float(d["rhoConst"]) == float(d["rhoNil"]) == C.RHO0 and float(d["rhoConstFresh"]) == C.RHO_FW
    assert float(d["HeatCapacity_Cp"]) == C.CP and float(d["gravity"]) == C.G
    assert float(d["f0"]) == pytest.approx(C.F0, rel=1e-6)
    assert float(d["zRoughBot"]) == C.Z0
    assert float(d["deltaT"]) == C.DT and int(d["nTimeSteps"]) == C.n_steps()
    assert d["delR"] == f"{C.N}*1.0"
    for k in ("viscAh", "viscAr", "diffKhT", "diffKrT", "diffKhS", "diffKrS"):
        assert float(d[k]) == 0.0
    e = _nml(MIT_IN / "data.exf")
    assert float(e["exf_albedo"]) == C.ALBEDO and float(e["ocean_emissivity"]) == C.EMISSIVITY
    assert float(e["hu"]) == 10.0 and float(e["ht"]) == 2.0 and e["useRelativeWind"] == ".TRUE."
    k = _nml(MIT_IN / "data.kpp")
    for name in ("difm0", "difs0", "dift0", "difmcon", "difscon", "diftcon"):
        assert float(k[name]) == 0.0
    assert "startDate_1 = 20170701" in (MIT_IN / "data.cal").read_text()
    sh = (HERE / "mitgcm" / "run_mitgcm.sh").read_text()
    assert f"jwtype={C.JERLOV_INDEX}" in sh and "EXF_CALC_ATMRHO" in sh and "ALLOW_BULK_LARGEYEAGER04" in sh


def test_croco_inputs_match_cases():
    txt = (HERE / "croco" / "croco.in").read_text().splitlines()
    val = lambda key, off=1: txt[next(i for i, s in enumerate(txt) if s.startswith(key)) + off].split()  # noqa
    nt, dt, _, _ = val("time_stepping")
    assert int(nt) == C.n_steps() and float(dt) == C.DT
    assert float(val("rho0:")[0].replace("d", "e")) == C.RHO0
    R0, T0, S0, tc, sc = (float(x) for x in val("lin_EOS_cff"))
    assert 1000.0 + R0 == C.RHO0 and T0 == C.T_REF and S0 == C.S_REF
    assert tc == pytest.approx(C.RHO0 * C.ALPHA) and sc == pytest.approx(C.RHO0 * C.BETA)
    assert [float(x.replace("d", "e")) for x in val("vertical_mixing")] == [0.0, 0.0, 0.0]
    assert float(val("bottom_drag")[2].replace("d", "e")) == C.Z0
    sh = (HERE / "croco" / "build_croco.sh").read_text()
    assert "N=36" in sh and "depth=36." in sh and f"lmd_Jwt={C.JERLOV_INDEX}" in sh
    f0 = re.search(r"f0=([0-9.eEdD+-]+)", sh).group(1).replace("d", "e").replace("D", "e")
    assert float(f0) == pytest.approx(C.F0, rel=1e-6)
    assert "LMD_SKPP" in sh and "LMD_NONLOCAL" in sh and "BULK_FLUX" in sh and "BULK_LW" in sh


def test_pyocean_configuration():
    cfg = C.RUNS_PYOCEAN
    assert cfg["ly04"].scheme == "ncar_ly04" and cfg["coare"].scheme == "coare"
    assert cfg["coare_croco"].relative_wind is False and cfg["coare_croco"].h_sfc == 0.0
    for k in C.BULK_RUNS:                                               # h_mix only where there is no wind feedback
        assert (cfg[k].h_mix > 0) == (not cfg[k].relative_wind)
    assert cfg["flux_croco_bg"].croco_background and cfg["flux_mitgcm"].prescribed == "mitgcm"
    assert C.JERLOV == "III" and C.JERLOV_INDEX == 5


# ------------------------------------------------------------------------------------------------ conversions
def test_croco_unit_conversions():
    s = dict(u10=np.array([3.0]), v10=np.array([4.0]), t2m=np.array([293.15]), q2m=np.array([0.012]),
             precip=np.array([1e-7]), lw_down=np.array([380.0]), sw_down=np.array([500.0]))
    f = forcing.croco_fields(s)
    assert f["wspd"][0] == 5.0 and f["tair"][0] == pytest.approx(20.0)
    assert f["rhum"][0] == pytest.approx(12.0)                     # g/kg ≥ 2: read as q (bulk_flux.F:769-774)
    assert f["prate"][0] == pytest.approx(1e-7 * 100 * 86400)      # cm/day
    assert f["radsw"][0] == pytest.approx((1 - C.ALBEDO) * 500.0)


def test_croco_init_header(tmp_path):
    p = tmp_path / "init.h"
    forcing.croco_init_header(p)
    lines = p.read_text().splitlines()
    T, S = C.initial_TS()
    assert sum("itemp" in s for s in lines) == 2 * C.N and sum("isalt" in s for s in lines) == 2 * C.N
    assert f"t(i,j,1,1,itemp)={T[0]:.10f}d0" in p.read_text()
    assert f"t(i,j,{C.N},2,isalt)={S[-1]:.10f}d0" in p.read_text()
    assert max(len(s) for s in lines if not s.startswith("!")) <= 72     # fixed-form Fortran


def test_mld_metric():
    T, S = C.initial_TS()
    prof = np.tile(T, (2, 1)), np.tile(S, (2, 1))
    prof[0][1, -10:] = T[-1]                                         # top 10 layers mixed
    prof[1][1, -10:] = S[-1]
    m = compare.mld(*prof)
    assert m[1] > 9.5 and m[0] < m[1]
    mixed = compare.mld(np.full((1, C.N), 20.0), np.full((1, C.N), 31.0))
    assert mixed[0] == C.DEPTH


# ------------------------------------------------------------------------------------------------ outputs
def _need(*paths: Path):
    for p in paths:
        if not p.exists():
            pytest.skip(f"{p} not on disk")


def test_series_is_era5_at_the_point():
    p = C.RUNS / f"series_{C.DAYS}d.npz"
    _need(p)
    s = np.load(p)
    assert s["time"].size == C.DAYS * 24 + 1
    for k in ("u10", "v10", "t2m", "q2m", "msl", "sw_down", "lw_down", "precip"):
        assert np.isfinite(s[k]).all(), k
    assert 285 < s["t2m"].mean() < 300 and 0.005 < s["q2m"].mean() < 0.02 and s["sw_down"].min() >= 0


def test_runs_read_and_compare():
    _need(C.RUNS / f"mitgcm_{C.DAYS}d", C.RUNS / f"croco_{C.DAYS}d" / "scm_avg.nc",
          C.RUNS / f"pyocean_{C.DAYS}d" / "ly04.npz", C.RUNS / f"pyocean_{C.DAYS}d" / "coare.npz")
    runs = compare.load_all()
    for k in ("mitgcm", "croco", "ly04", "coare"):
        r = runs[k]
        assert r["t_mean"].size == C.DAYS * 24, k
        assert np.isfinite(r["sst"]).all() and np.isfinite(r["q_net"]).all(), k
    for k in C.RUNS_PYOCEAN:
        if k in runs:                                                   # the 5 × 5 column stays 1-D (README §5)
            assert runs[k]["spread"] < 1e-3, (k, runs[k]["spread"])
    # same shortwave input in the three models (same albedo, same ERA5 series)
    sw = {k: compare.means(runs[k])["q_sw"] for k in ("mitgcm", "croco", "ly04", "coare")}
    assert max(sw.values()) - min(sw.values()) < 1e-3 * sw["mitgcm"]
    assert "Q_net" in compare.table(runs) and "nan" not in compare.state_table(runs)
