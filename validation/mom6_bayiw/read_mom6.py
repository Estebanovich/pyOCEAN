"""
MOM6 output of the BayIW benchmark in pyOCEAN's formats (spec 2026-10-13-mom6-bayiw-benchmark-design §3):
section series as validation/bayiw_2d/run_pyocean.record (so bayiw_2d/compare.report compares MOM6 unchanged),
and the FMS clock table of stdout for the cost comparison.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import netCDF4
import numpy as np

from core.vertical.ale import face_transport_thickness
from validation.bayiw_2d import compare, section
from validation.bayiw_baroclinic import bayiw

COL = 1                                                   # the section column (land | column | land …)


def section_series(run_dir: Path, case: str, bay: str) -> dict:
    """η, the T and V sections on the MITgcm levels, T110/T250/V110 and heat per record, as run_pyocean.record:
    MOM6 layers (surface first) flipped to bed first, conservative PPM_H4 sampling (section.to_mitgcm_levels),
    v-face layers = the mean of the two clipped upwind thicknesses (core.vertical.ale.face_transport_thickness),
    the face bed the shallower side."""
    s = section.section_inputs(case, bay)
    H = np.asarray(s["H"][:, 0], float)
    Hc = np.nan_to_num(H)
    act = np.isfinite(H)
    Hf = np.concatenate([Hc[:1], np.minimum(Hc[1:], Hc[:-1]), Hc[-1:]])
    area = float(s["dx"][0]) * np.asarray(s["dy"], float)
    out = {k: [] for k in ("t_h", "eta", "T110", "T250", "V110", "T_sec", "V_sec", "n_trunc", "heat")}
    with netCDF4.Dataset(Path(run_dir) / "prog.nc") as ds:
        t_days = np.asarray(ds["Time"][:], float)
        for n in range(t_days.size):
            h = np.asarray(ds["h"][n, :, :, COL], float)[::-1][:, :, None]              # (N, ny, 1) bed first
            T = np.asarray(ds["temp"][n, :, :, COL].filled(np.nan), float)[::-1][:, :, None]
            v = np.asarray(ds["v"][n, :, :, COL].filled(0.0), float)[::-1][:, :, None]  # (N, ny+1, 1)
            Tl = section.to_mitgcm_levels(h, T, H[:, None])[::-1, :, 0]
            qn, qs = face_transport_thickness(h.swapaxes(1, 2), Hc[None, :], act[None, :], True)
            hv = (0.5 * (qn + qs)).swapaxes(1, 2)
            Vl = section.to_mitgcm_levels(hv, v, Hf[:, None])[::-1, :, 0]
            lev = dict(T=np.nan_to_num(Tl)[None], V=np.nan_to_num(Vl)[None], zc=bayiw.Z_C)
            out["t_h"].append(t_days[n] * 24.0)
            out["eta"].append(np.asarray(ds["SSH"][n, :, COL].filled(0.0), float))
            out["T110"].append(compare.mit_at_depth(lev, H, "T", bayiw.Z110)[0])
            out["T250"].append(compare.mit_at_depth(lev, H, "T", bayiw.Z250)[0])
            out["V110"].append(compare.mit_at_depth(lev, H, "V", bayiw.Z110)[0])
            out["T_sec"].append(Tl)
            out["V_sec"].append(Vl)
            out["n_trunc"].append(0)
            out["heat"].append(float(np.nansum(h[:, :, 0] * np.nan_to_num(T[:, :, 0]) * area[None])))
    res = {k: np.array(v) for k, v in out.items()}
    res["rpe"] = np.full(2, np.nan)
    res["meta_json"] = json.dumps(dict(model="MOM6", case=case, bay=bay, run_dir=str(run_dir)))
    return res


def save_series(series: dict, case: str, bay: str, coord: str) -> Path:
    """section.RUNS/pyocean/<case>_<bay>_mom6<coord>/series.npz, read by bayiw_2d/compare.report(case, 'mom6'+coord)."""
    out = section.RUNS / "pyocean" / f"{case}_{bay}_mom6{coord}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "series.npz", **series)
    return out
