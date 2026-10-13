"""
MOM6 inputs for the BayIW benchmark (spec 2026-10-13-mom6-bayiw-benchmark-design §2), from validation.bayiw_baroclinic:

- ``ocean_hgrid.nc``: mosaic supergrid of the stretched Cartesian grid (half-cell lengths, quarter-cell areas;
  MOM_grid_initialize.F90:192-289); x, y are labels in km;
- ``topog.nc``: depth positive down, 0 on land (MITgcm effective depth, bayiw.mitgcm_depth);
- ``ts_z.nc``: T on z-levels whose centres have the MITgcm interfaces as midpoints, because MOM6 builds the source
  cell edges as midpoints of the centres (MOM_horizontal_regridding.F90:454-480); S constant; _FillValue mandatory;
- ``wind.nc``: C-grid stress, STRESS_Y on the north face of each row (MOM_surface_forcing.F90:773-787);
- ``hyb.nc`` (HYCOM1: interface densities and the z* floor) and ``layers.nc`` (HYBGEN: layer densities), with the
  bayiw targets in the linear-EOS density units of the model (MOM6 evaluates the linear EOS without pressure).
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from validation.bayiw_2d import section
from validation.bayiw_baroclinic import bayiw
from validation.bayiw_cut import cut

FILL = -1.0e20
S0 = 35.0                                               # constant salinity (DRHO_DS = 0)
NX_SECTION = 10                                         # land | section column | 8 land (FMS halos)


def source_centres() -> np.ndarray:
    """Centres c_k [m, positive down] with (c_k + c_{k+1})/2 = the MITgcm interfaces."""
    zf = -bayiw.Z_F
    c = np.empty(zf.size - 1)
    c[0] = 0.5 * zf[1]
    for k in range(c.size - 1):
        c[k + 1] = 2.0 * zf[k + 1] - c[k]
    return c


def domain(kind: str, case: str, bay: str) -> dict:
    """dx (nx,), dy (ny,) [m], H (ny, nx) NaN land, T (50, ny, nx), wind (nrec, ny, nx) on the south faces, x0, y0."""
    if kind == "section":
        # MITgcm's wall | column i=280 | wall, padded with land to NX_SECTION columns: FMS checks every halo
        # (also the south/north ones and the wider supergrid and barotropic halos) against the x size
        # (mpp_domains_define.inc:899-906), so three columns cannot run; land columns change nothing
        s = section.section_inputs(case, bay)
        rep = lambda a: np.repeat(a, NX_SECTION, axis=-1)                          # noqa: E731
        H = rep(s["H"]).astype(float)
        H[:, [0] + list(range(2, NX_SECTION))] = np.nan
        return dict(dx=np.repeat(s["dx"], NX_SECTION), dy=s["dy"], H=H, T=rep(s["T"]), wind=rep(s["wind"]),
                    x0=s["x0"] - float(s["dx"][0]), y0=bayiw.Y0)
    if kind == "cut":
        d = cut.cut_inputs(case, bay, "py")
        return dict(dx=d["dx"], dy=d["dy"], H=d["H"], T=d["T"], wind=d["wind"], x0=d["x0"], y0=d["y0"])
    raise ValueError(f"unknown kind {kind!r}")


def targets(case: str, T: np.ndarray, H: np.ndarray) -> np.ndarray:
    """The hybrid interface densities (26,) of bayiw.build_model [kg m⁻³]."""
    rho = bayiw.RHO0 - bayiw.T_COEF * (T[:, np.isfinite(H)] - bayiw.CASES[case]["t_ref"])
    return (bayiw.centred_targets(float(rho.min()), float(rho.max()), 25, bayiw.N_LIGHT) if case == "2layer"
            else bayiw.profile_targets(rho, 25, bayiw.N_LIGHT))


def _write_hgrid(path: Path, dx: np.ndarray, dy: np.ndarray, x0: float, y0: float) -> None:
    nx, ny = dx.size, dy.size
    hx, hy = np.repeat(0.5 * dx, 2), np.repeat(0.5 * dy, 2)                       # supergrid half cells
    xs = (x0 + np.concatenate([[0.0], np.cumsum(hx)])) / 1e3
    ys = (y0 + np.concatenate([[0.0], np.cumsum(hy)])) / 1e3
    with netCDF4.Dataset(path, "w") as ds:
        for n, s in (("nxp", 2 * nx + 1), ("nyp", 2 * ny + 1), ("nx", 2 * nx), ("ny", 2 * ny)):
            ds.createDimension(n, s)
        v = ds.createVariable("x", "f8", ("nyp", "nxp"))
        v[:] = np.broadcast_to(xs[None], (2 * ny + 1, 2 * nx + 1))
        v.units = "km"
        v = ds.createVariable("y", "f8", ("nyp", "nxp"))
        v[:] = np.broadcast_to(ys[:, None], (2 * ny + 1, 2 * nx + 1))
        v.units = "km"
        v = ds.createVariable("dx", "f8", ("nyp", "nx"))
        v[:] = np.broadcast_to(hx[None], (2 * ny + 1, 2 * nx))
        v.units = "m"
        v = ds.createVariable("dy", "f8", ("ny", "nxp"))
        v[:] = np.broadcast_to(hy[:, None], (2 * ny, 2 * nx + 1))
        v.units = "m"
        v = ds.createVariable("area", "f8", ("ny", "nx"))
        v[:] = hy[:, None] * hx[None]
        v.units = "m2"
        ds.createVariable("angle_dx", "f8", ("nyp", "nxp"))[:] = 0.0


def _write_topog(path: Path, H: np.ndarray) -> None:
    ny, nx = H.shape
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("ny", ny)
        ds.createDimension("nx", nx)
        v = ds.createVariable("depth", "f8", ("ny", "nx"))
        v[:] = np.nan_to_num(H)
        v.units = "m"


def _write_ts(path: Path, T: np.ndarray, H: np.ndarray, xc: np.ndarray, yc: np.ndarray) -> None:
    """T on the source levels; lon/lat = the model cell-centre labels [km] (TEMP_SALT_INIT_VERTICAL_REMAP_ONLY:
    the data are already on the model grid, only the vertical remap is done)."""
    nz, ny, nx = T.shape
    c = source_centres()
    Tf = np.where(np.isfinite(T) & np.isfinite(H)[None], T, FILL)
    with netCDF4.Dataset(path, "w") as ds:
        for n, s in (("Time", None), ("depth", nz), ("lat", ny), ("lon", nx)):
            ds.createDimension(n, s)
        ds.createVariable("Time", "f8", ("Time",))[:] = [0.0]
        ds.createVariable("depth", "f8", ("depth",))[:] = c
        ds["depth"].units = "m"
        ds["depth"].positive = "down"
        ds.createVariable("lat", "f8", ("lat",))[:] = yc
        ds.createVariable("lon", "f8", ("lon",))[:] = xc
        for name, val in (("ptemp", Tf), ("salt", np.where(Tf == FILL, FILL, S0))):
            v = ds.createVariable(name, "f8", ("Time", "depth", "lat", "lon"), fill_value=FILL)
            v[0] = val


def _write_wind(path: Path, wind: np.ndarray) -> None:
    nrec, ny, nx = wind.shape
    ty = np.zeros((nrec, ny, nx))
    ty[:, :-1] = wind[:, 1:]                       # MOM6 row j = north face of cell j = MITgcm v-row j + 1
    with netCDF4.Dataset(path, "w") as ds:
        for n, s in (("Time", None), ("lat", ny), ("lon", nx)):
            ds.createDimension(n, s)
        t = ds.createVariable("Time", "f8", ("Time",))
        t[:] = (np.arange(nrec) + 0.5) * 1200.0 / 86400.0
        t.units = "days since 0001-01-01 00:00:00"
        for name, val in (("STRESS_X", np.zeros_like(ty)), ("STRESS_Y", ty)):
            v = ds.createVariable(name, "f8", ("Time", "lat", "lon"))
            v[:] = val
            v.units = "Pa"


def _write_coords(path_hyb: Path, path_layers: Path, case: str, T: np.ndarray, H: np.ndarray) -> None:
    tgt = targets(case, T, H)
    with netCDF4.Dataset(path_hyb, "w") as ds:
        ds.createDimension("interfaces", tgt.size)
        ds.createDimension("layers", tgt.size - 1)
        ds.createVariable("sigma2", "f8", ("interfaces",))[:] = tgt
        ds.createVariable("dz", "f8", ("layers",))[:] = bayiw.hybrid_dz_top(25)
    with netCDF4.Dataset(path_layers, "w") as ds:
        ds.createDimension("layers", tgt.size - 1)
        ds.createVariable("Layer", "f8", ("layers",))[:] = 0.5 * (tgt[1:] + tgt[:-1])


def write_inputs(kind: str, case: str, bay: str, out_dir: Path) -> None:
    """All MOM6 input files of one case in out_dir/INPUT."""
    d = domain(kind, case, bay)
    inp = Path(out_dir) / "INPUT"
    inp.mkdir(parents=True, exist_ok=True)
    _write_hgrid(inp / "ocean_hgrid.nc", d["dx"], d["dy"], d["x0"], d["y0"])
    _write_topog(inp / "topog.nc", d["H"])
    xc = (d["x0"] + np.cumsum(d["dx"]) - 0.5 * d["dx"]) / 1e3
    yc = (d["y0"] + np.cumsum(d["dy"]) - 0.5 * d["dy"]) / 1e3
    _write_ts(inp / "ts_z.nc", d["T"], d["H"], xc, yc)
    _write_wind(inp / "wind.nc", d["wind"])
    _write_coords(inp / "hyb.nc", inp / "layers.nc", case, d["T"], d["H"])
