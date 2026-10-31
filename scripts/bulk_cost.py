#!/usr/bin/env python
"""
Cost of the bulk surface fluxes (plan docs/superpowers/plans/2026-10-31-n2b-bulk-surface-fluxes.md, task 6; spec
criterion 8: ratio <= 1.05).  Timing protocol of ``scripts/bbl_cost.py``: median wall time per step of 50 steps
after 10 warm-up steps, 5 threads, one process per mode (``--one``), modes alternating (off, on, off, on, ...),
3 rounds.

Case: BayIW cut, z* on the 50 MITgcm levels (``validation/bayiw_cut/cut.py`` ``build_model``, same arguments),
Delta t 120 s, with ``salinity=True`` (S = 35 g/kg uniform, s_coef 0.78) in both modes:

* ``on``:  ``surface_fluxes=BulkFluxes(provider, scheme='ncar')`` (``core.components.surface_fluxes``), the
  provider a real ``forcing.atmosphere.AtmosphereProvider`` over a synthetic canonical atmosphere with hourly
  records on a regular lon/lat grid of the size of the model grid (one source point per model cell, plus a margin,
  with a land block of NaN so ``fill_land`` works), so each new record costs a full land fill + bilinear
  interpolation to every T point.  With Delta t = 120 s a record is read every 30 steps: the median step does not
  read, the mean step does; both are reported.
* ``off``: the same model without ``surface_fluxes``; ``tau_x``, ``tau_y``, ``fw`` and ``heat`` prescribed
  constant every step, taken from one call of the bulk component on the initial state (same magnitudes).

The script checks that the bulk term is active (non-zero latent heat and stress) and reports per step (sum over the
timed steps / 50): the whole component (``SurfaceFluxes.forcing``), the provider call (``AtmosphereProvider.__call__``:
time interpolation of the two cached records, plus reads), the record reads (``AtmosphereProvider._record`` on a
cache miss: ``isel`` + ``load`` + ``fill_land`` + ``to_model_grid``) and the number of records read.

    python -I scripts/bulk_cost.py [--rounds 3]

Run alone, with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NWARM, NMEAS = 10, 50
DT = 120.0
HOUR = 3600.0
ORIGIN = np.datetime64("2017-07-01T00:00", "ns")
LON0, LAT0 = -111.0, 28.5                       # place of the Cartesian cut on the sphere (f0 = 6.97e-5 → 28.6 °N)
SAL = dict(salinity=True, s_coef=0.78, s_ref=35.0)


def _lonlat(hg: xr.Dataset) -> xr.Dataset:
    """The Cartesian cut grid with lon_t, lat_t [°] (equirectangular around LON0, LAT0) for the provider."""
    hg = hg.copy()
    m_per_deg = 111320.0
    hg["lon_t"] = LON0 + (hg["x_t"] - float(hg["x_t"].min())) / (m_per_deg * np.cos(np.deg2rad(LAT0)))
    hg["lat_t"] = LAT0 + (hg["y_t"] - float(hg["y_t"].min())) / m_per_deg
    return hg


def _source(hg: xr.Dataset, nrec: int) -> xr.Dataset:
    """
    Synthetic canonical atmosphere (time, lat, lon), hourly from t = 0, on a regular grid of the size of the model
    grid (+ 4 points of margin each side) covering lon_t/lat_t; realistic summer values varying in space and time;
    the western 10 % of the source is land (NaN) so the provider's sea-over-land fill has work to do.
    """
    lon_t, lat_t = hg["lon_t"].values, hg["lat_t"].values
    ny, nx = lon_t.shape
    dlon = (lon_t.max() - lon_t.min()) / (nx - 1)
    dlat = (lat_t.max() - lat_t.min()) / (ny - 1)
    lon = lon_t.min() + dlon * np.arange(-4, nx + 4)
    lat = (lat_t.min() + dlat * np.arange(-4, ny + 4))[::-1]
    tt = HOUR * np.arange(nrec, dtype=float)
    ph = np.sin(2 * np.pi * tt / 86400.0)[:, None, None]
    x = ((lon - lon.min()) / (lon.max() - lon.min()))[None, None, :]
    y = ((lat - lat.min()) / (lat.max() - lat.min()))[None, :, None]
    one = np.ones((tt.size, lat.size, lon.size))
    f = {"u10": 3.0 + 4.0 * x + 1.0 * ph, "v10": 6.0 - 2.0 * y - 0.5 * ph, "t2m": 296.0 + 2.0 * y + 0.8 * ph,
         "q2m": 0.014 + 0.002 * x, "msl": 101200.0 + 200.0 * y, "sw_down": np.maximum(0.0, 500.0 + 450.0 * ph),
         "lw_down": 400.0 + 10.0 * y, "precip": 1e-7 * (1.0 + x) * (1.0 + 0.5 * ph)}
    land = np.zeros((lat.size, lon.size), bool)
    land[:, : max(1, lon.size // 10)] = True
    data = {k: (("time", "lat", "lon"), np.where(land[None], np.nan, np.asarray(v, float) * one)) for k, v in f.items()}
    return xr.Dataset(data, coords={"time": ORIGIN + (tt * 1e9).astype("timedelta64[ns]"), "lat": lat, "lon": lon})


def _build(bulk: bool):
    """BayIW cut z* as ``cut.build_model('2layer', 'bay', 'zstar', 120)``, with salinity; bulk fluxes if ``bulk``."""
    from core.components.surface_fluxes import BulkFluxes, SurfaceFluxes
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    from core.grid.horizontal import create_cartesian_grid_stretched
    from forcing.atmosphere import AtmosphereProvider
    from validation.bayiw_baroclinic import bayiw
    from validation.bayiw_cut import cut
    d = cut.cut_inputs("2layer", "bay", "py")
    hg = create_cartesian_grid_stretched(d["dx"], d["dy"], x0=d["x0"], y0=d["y0"], f0=bayiw.F0)
    nrec = int(np.ceil((NWARM + NMEAS + 1) * DT / HOUR)) + 2
    atm = AtmosphereProvider(_source(_lonlat(hg), nrec), _lonlat(hg), ORIGIN)
    cfg = BulkFluxes(atm, scheme="ncar")
    kw = dict(surface_fluxes=cfg) if bulk else {}
    m = SigmaBaroclinic(hg, d["H"], dt=DT, n_layers=bayiw.DELR.size, rho0=bayiw.RHO0, t_coef=bayiw.T_COEF,
                        t_ref=bayiw.CASES["2layer"]["t_ref"], z0=None, closure="constant", nu_const=1e-5, ah=100.0,
                        kh=100.0, kv=1e-5, pgf="fv", rho_ref="auto", vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=bayiw.DELR.copy()), open_boundaries=("west", "east", "south"),
                        orlanski_cmax=0.45, sponge_cells=30, sponge_tau=(1200.0, 60.0), sponge_tracer=False,
                        mix_diagnostic=False, **SAL, **kw)
    h_src, T_src = bayiw.mitgcm_source(d["H"], d["T"])
    st = m.initial_state(np.zeros(d["H"].shape), T_src, h_src=h_src, S=np.full(np.shape(T_src), 35.0))
    if bulk:
        return m, st, (lambda m, st, n: m.step(st)), atm
    sf = SurfaceFluxes(cfg, fs=m.fs, area=m.area, ds=m.ds, ale=m.ale, periodic=m.periodic, rho0=m.rho0,
                       salinity=True)
    tx, ty, fw, heat = sf.forcing(st, 0.5 * DT)                     # constant prescribed forcing, bulk magnitudes
    return m, st, (lambda m, st, n: m.step(st, tau_x=tx, tau_y=ty, fw=fw, heat=heat)), atm


def _wrap(obj, name: str, acc: list) -> None:
    """Time every call of the bound method ``obj.name`` (instance wrapper)."""
    orig = getattr(obj, name)

    def timed(*a, **k):
        t0 = time.perf_counter()
        out = orig(*a, **k)
        acc.append(time.perf_counter() - t0)
        return out
    setattr(obj, name, timed)


class _TimedProvider:
    """Proxy of an ``AtmosphereProvider`` that times each call (``coverage`` and ``origin`` passed through)."""

    def __init__(self, atm, acc: list) -> None:
        self.atm, self.acc = atm, acc
        self.origin = atm.origin

    def coverage(self) -> tuple[float, float]:
        return self.atm.coverage()

    def __call__(self, t: float) -> dict:
        t0 = time.perf_counter()
        out = self.atm(t)
        self.acc.append(time.perf_counter() - t0)
        return out


def measure(bulk: bool) -> dict:
    """Median and mean s/step; per-step time of the component, the provider call and the record reads."""
    from core.dynamics import sigma_baroclinic as sb
    sb.set_threads(5)
    m, st, step, atm = _build(bulk)
    comp, call, reads = [], [], []
    if bulk:
        _wrap(m._surface_fluxes, "forcing", comp)
        assert m._surface_fluxes.cfg.atmosphere is atm
        m._surface_fluxes.cfg.atmosphere = _TimedProvider(atm, call)   # ``atm(t)`` looks __call__ up on the type
        orig_record = atm._record

        def record(n):                               # time cache misses only (a hit is a dict lookup)
            miss = n not in atm._cache
            t0 = time.perf_counter()
            out = orig_record(n)
            if miss:
                reads.append(time.perf_counter() - t0)
            return out
        atm._record = record
    for n in range(NWARM):
        st = step(m, st, n)
    for a in (comp, call, reads):
        a.clear()
    t = []
    qlat = tau = 0.0
    for n in range(NWARM, NWARM + NMEAS):
        t0 = time.perf_counter()
        st = step(m, st, n)
        t.append(time.perf_counter() - t0)
        if bulk:
            qlat = max(qlat, float(np.max(np.abs(m._surface_fluxes.last["q_lat"]))))
            tau = max(tau, float(np.max(np.hypot(m._surface_fluxes.last["tau_x"], m._surface_fluxes.last["tau_y"]))))
    assert np.all(np.isfinite(st.eta)), "eta not finite"
    if bulk:
        assert len(comp) == NMEAS, f"bulk component ran {len(comp)} times in {NMEAS} steps"
        assert qlat > 0.0 and tau > 0.0, f"bulk term not active: max|Q_lat| {qlat}, max|tau| {tau}"
    per_step = lambda a: float(sum(a)) / NMEAS                         # noqa: E731
    return dict(step=float(np.median(t)), mean=float(np.mean(t)), comp=per_step(comp), call=per_step(call),
                reads=per_step(reads), nreads=len(reads), read_each=float(np.mean(reads)) if reads else 0.0,
                qlat=qlat, tau=tau)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--one", choices=("off", "on"), help="internal: one mode, JSON on stdout")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    if a.one:
        print("RESULT " + json.dumps(measure(a.one == "on")))
        return
    name = "bayiw_cut_zstar_salinity"
    res = {"off": [], "on": []}
    for _ in range(a.rounds):
        for mode in ("off", "on"):
            out = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--one", mode],
                                 capture_output=True, text=True, check=True, cwd=ROOT).stdout
            res[mode].append(json.loads(out.split("RESULT ", 1)[1]))
            r = res[mode][-1]
            print(f"  {name} {mode:3s} median {r['step']:.4f} mean {r['mean']:.4f} s/step  component "
                  f"{r['comp'] * 1e3:6.2f} ms  provider call {r['call'] * 1e3:6.2f} ms  "
                  f"reads {r['reads'] * 1e3:6.2f} ms"
                  f" ({r['nreads']} records, {r['read_each'] * 1e3:.1f} ms each)", flush=True)
    ratios = [on["step"] / off["step"] for on, off in zip(res["on"], res["off"])]
    mratios = [on["mean"] / off["mean"] for on, off in zip(res["on"], res["off"])]
    med = lambda mode, key: float(np.median([r[key] for r in res[mode]]))   # noqa: E731
    t0, t1 = med("off", "step"), med("on", "step")
    print(f"{name}  off {t0:.4f}  on {t1:.4f} s/step (median)  ratio min/median/max "
          f"{min(ratios):.3f}/{np.median(ratios):.3f}/{max(ratios):.3f};  mean-step ratio min/median/max "
          f"{min(mratios):.3f}/{np.median(mratios):.3f}/{max(mratios):.3f};  component {med('on', 'comp') * 1e3:.1f} ms"
          f" = {100 * med('on', 'comp') / t1:.1f} % of the median step; provider call {med('on', 'call') * 1e3:.1f} ms,"
          f" of which record reads {med('on', 'reads') * 1e3:.1f} ms/step ({med('on', 'read_each') * 1e3:.0f} ms per"
          f" record, {med('on', 'nreads'):.0f} records in {NMEAS} steps)", flush=True)
    print("RATIOS " + json.dumps(dict(median=ratios, mean=mratios, res=res)))


if __name__ == "__main__":
    main()
