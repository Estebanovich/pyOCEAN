# CEFI preprocessing tools for NWA12: reference read for the pyOCEAN regional chain

Source: `~/pyOCEAN_refs/CEFI-regional-MOM6`, git `36b8f49`; `tools/` paths relative to
`~/pyOCEAN_refs/CEFI-regional-MOM6/tools/`, NWA12 run files relative to `exps/NWA12.COBALT/`.
Read 2026-11-01 for the regional chain (spec §"Entradas" of the MOM6 run, plan Task 10
`mom6_inputs.py`). The tools are written for GFDL's PPAN/Gaea (hard-coded `/work/acr/...`,
`/archive/...`, `$TMPDIR`, `dmget`, `gcp`, `module load`), so on this Mac every path must come from
the YAML and the HPC-only commands (`dmget`, `gcp`) must be dropped. MOM6 line references use
`~/pyOCEAN_refs/MOM6` git `0e9a561`.

Python dependencies across the tools: `xarray`, `numpy`, `pandas`, `pyyaml`, **`xesmf`** (ESMF
regridding), **`HCtFlood`** (`from HCtFlood import kara`, land flooding;
`boundary/boundary.py:67-101` adds a hard-coded path `/home/Andrew.C.Ross/git/HCtFlood` to `sys.path`,
`write_glorys_initial.py:16` imports it directly). Command-line tools: **NCO** (`ncrcat`, `ncks`,
`ncpdq`, `ncap2`), **CDO** (`setmisstonn`, `sub`, `setrtoc`, `chname`, `expr`, `merge`),
`copernicusmarine` (GLORYS download) and **FRE-NCtools** (grid only, §6).

---

## 1. Open-boundary segments from GLORYS: `boundary/write_glorys_boundary.py`

**Command:** `./write_glorys_boundary.py --config glorys_obc.yaml` (`:8-9, 104-115`).

**Config** (`boundary/glorys_obc.yaml:1-23`): `first_year`, `last_year`, `glorys_dir`, `output_dir`,
`hgrid` (`ocean_hgrid.nc`), `ncrcat_years` (bool), `ncrcat_names`, `segments` (list of
`{id, border}`; NWA12: 1 south, 2 north, 3 east), `variables` (`thetao`, `so`, `zos`, `uv`).

**Input:** daily GLORYS files `glorys_dir/GLORYS_REANALYSIS_{year}-*.nc` (`:29-32`) with `latitude`,
`longitude`, `depth`, `thetao`, `so`, `zos`, `uo`, `vo`, **already flooded over land** ("Have to make
sure nan values in Glorys have been filled", `:4`; done with `cdo setmisstonn` in
`boundary/fill_glorys_nn.sh:32`). The first record of the first year is floored to midnight and the
last record of the last year ceiled (`:34-40`) so that the files span whole days.

**Processing** (`boundary/boundary.py`, class `Segment`, `:293-926`):
- Segment coordinates are the **supergrid** nodes of the edge row/column: south `nyp=0`, north
  `nyp=-1`, west `nxp=0`, east `nxp=-1`, with `lon=x`, `lat=y`, `angle=angle_dx` (`:334-360`;
  `angle_dx` converted from degrees if its `units` attribute says so, `:319-325`). This is why
  NWA12 runs with `BRUSHCUTTER_MODE = True` (`mom6_obc_tides.md` §2).
- Horizontal interpolation: `xesmf` `nearest_s2d` onto the segment as a locstream (`reuse_regrid` `:278-291`, calls `:524-541, 657-665`).
- Velocities: `rotate_uv(u_east, v_north, angle_dx)` → model-relative
  (`urot = cos α u + sin α v`, `vrot = −sin α u + cos α v`, `:18-42, 570-579`), using **`angle_dx`**
  (counter-clockwise from east), not MOM6's corner angle (`mom6_grid.md` §6).
- Missing values filled along the segment (`bfill`) then downward (`ffill` in z), remaining NaN → 0
  (`fill_missing`, `:45-64`).
- Layer thicknesses for MOM6's remapping: `z_to_dz` puts interfaces half-way between GLORYS depth
  levels, the last one at `max_depth = 6500 m` (`:244-276`); written as `dz_u_segment_00n`,
  `dz_v_…`, `dz_thetao_…`, etc.; the `z` coordinate is replaced by `0..nz-1` (`:595-597, 680-682`).
- `zos` is treated as a 2-D tracer (`write_glorys_boundary.py:50-52`).

**Output** (per segment `n`, per year, then concatenated by `ncrcat` if `ncrcat_years`,
`:55-61, 96-102`): `uv_00n.nc` (`u_segment_00n`, `v_segment_00n`, `dz_u_…`, `dz_v_…`),
`thetao_00n.nc`, `so_00n.nc`, `zos_00n.nc`; variables `lon_segment_00n`, `lat_segment_00n`; dims
`time` (unlimited), `nz_segment_00n`, `ny_segment_00n`, `nx_segment_00n` (one of them length 1);
`NETCDF3_64BIT`, `_FillValue = 1e20`, gregorian calendar (`boundary.py:377-414`). These names match
`OBC_SEGMENT_00n_DATA` in NWA12 (`MOM_parameter_doc.all:633-637`).

Daily variant: `boundary/write_glorys_boundary_daily.py --config config.yaml --year Y --month M
--day D` (one day), or `--ncrcat_years [--adjust_timestamps]` (concatenate) (`:16-21, 177-182`),
driven by `boundary/glorys_obc_workflow/mom6_obc_workflow.sh` and its SLURM templates.

## 2. Tides on the segments: `boundary/write_tpxo_boundary.py`

**Command:** `./write_tpxo_boundary.py --config tpxo_obc.yaml` (`:5, 100-102`).

**Config** (`boundary/tpxo_obc.yaml:1-26`): `constituents` (indices 0–9 into TPXO9's list:
m2, s2, n2, k2, k1, o1, p1, q1, mm, mf, `:82-99`), `indices` (`ny_start 400, ny_end 1500,
nx_start 1600, nx_end -1`: a TPXO subset around NWA, `:126-138`), `segments`, `tpxo_dir`
(`/work/acr/tpxo9/`), `grid_file`, `output_dir`.

**Input:** **TPXO9 v1** (`h_tpxo9.v1.nc`, `u_tpxo9.v1.nc`; `:17-47`), not TPXO10. Elevation
`ha`, `hp` (m, degrees); velocities `ua`, `up`, `va`, `vp` with amplitude multiplied by 0.01
("convert to m/s", `:32, 43`), i.e. TPXO's **velocity** fields in cm/s, not the transports `Ua`,
`Va` [verificar the TPXO9 v1 variable units against the file header]. Land mask = amplitude ≤ 0 of
the first constituent (`:23, 34, 45`).

**Processing:** complex `h = ha·exp(−i hp)`, `u = ua·exp(−i up)` (`:22, 33, 44`); flood over land with
HCtFlood (constituent used as the "time" axis), `nearest_s2d` to the supergrid segment nodes, fill
missing (`boundary.py:708-788, 790-926`). Elevation → `zamp = |h|`, `zphase = −angle(h)` (= `hp`, in
radians; `:769-771`). Velocity: complex u, v → tidal ellipse (`ap2ep`, Xu 2000/2002 after Foreman's
convention, `:127-180`) → inclination reduced by `angle_dx` (`INC -= angle`, `:893-896`) → back to
amplitudes and phases (`ep2ap`, `:183-241`). The rotation is done on the ellipse because u and v
have different phases.

**Output:** `tz_00n.nc` (`zamp_segment_00n`, `zphase_segment_00n`) and `tu_00n.nc`
(`uamp_…`, `uphase_…`, `vamp_…`, `vphase_…`), dims `(time, constituent, ny|nx_segment, nx|ny_segment)`
with a single dummy time `1980-12-01` ("Seem to need a time dimension to have it read by MOM",
`:54-61`). Constant amplitudes; nodal modulation is applied by MOM6 (`mom6_obc_tides.md` §3).

Note a stated but harmless bug: the comment "TODO: BUG: should be vresource and vimsource"
(`boundary.py:829`) sits above lines that already use `vresource`/`vimsource`.

## 3. Initial state: `initial/write_glorys_initial.py`

**Command:** `./write_glorys_initial.py --config_file glorys_ic.yaml` (`:4-5, 144-159`).

**Config** (`initial/glorys_ic.yaml:1-13`): `glorys_file` (one daily GLORYS file, e.g.
`GLORYS_REANALYSIS_1993-01-01.nc`), `vgrid_file` (`vgrid_75_2m.nc`), `grid_file`
(`ocean_hgrid.nc`), `output_file`, `reuse_weights`, `variable_names`.

**Vertical grid:** `tools/grid/vgrid_75_2m.nc` holds one variable `dz(nz=75)`: 2 m at the top,
250 m at the bottom, sum 6500.0 m (read 2026-11-01). `vgrid_to_layers` returns layer centres from
interfaces `[0, cumsum(dz)]` with the last interface forced to 6500 m (`initial/depths.py:4-18`).

**Processing** (`:30-141`): time floored to midnight (`:58`); **vertical first**: linear
interpolation of GLORYS in depth onto the layer centres, extrapolating below GLORYS's deepest level
and then `ffill` downward (`:62`); flood T, S, u, v over land with HCtFlood per level and `zos`
separately (`:65-70`); **horizontal**: `nearest_s2d` to the T points (`x[1::2,1::2]`) for T, S, SSH,
and to **every supergrid node** for u, v, which are then rotated with `angle_dx` and subsampled to
u points `[1::2, 0::2]` and v points `[0::2, 1::2]` (`:73-100`).

**Output:** `temp`, `salt`, `ssh`, `u`, `v` on `(time, zl, yh|yq, xh|xq)`, `zl` "Layer pseudo-depth,
-z*" (`:102-141`), `NETCDF3_64BIT`, no `_FillValue`.

**How NWA12 reads it** (`exps/NWA12.COBALT/INPUT/MOM_input`): `THICKNESS_CONFIG = "coord"` (`:675`,
layers = the 75 z* target thicknesses), `TS_CONFIG = "file"` with `TS_FILE` = this file (`:706, 726`,
values taken layer by layer, no remap), `DEPRESS_INITIAL_SURFACE = True` with
`SURFACE_HEIGHT_IC_FILE` = this file, variable `ssh` (`:734, 741-745`), `VELOCITY_CONFIG = "file"`,
`u`, `v` (`:751-767`), `ALE_COORDINATE_CONFIG = "FILE:vgrid_75_2m.nc,dz"` (`:573`). The run in the
repository restarts from `INPUT/MOM.res*.nc` (`input.nml_24hr:3-5`), so these only matter for a cold
start. Because the T/S file is on the z\* centres with the bottom value extended, MOM6 relies on its
own ALE initial regrid/remap to cut layers at the bathymetry (not read here).

GLORYS download: `initial/get_glorys_data.sh -u USER -p PASS -o OUTDIR -x LONMIN -X LONMAX -y LATMIN
-Y LATMAX -s YYYY-MM-DD -e YYYY-MM-DD` → `copernicusmarine subset -i
cmems_mod_glo_phy_my_0.083deg_P1D-m -v so thetao uo vo zos …` one file per day
(`:4, 46-63`).

## 4. Atmosphere: `tools/atmos/*`

The ERA5 fields reach the model through FMS `data_override` (`exps/NWA12.COBALT/data_table:1-22`),
not through MOM6. The tools only prepare yearly, regionally sliced, unpacked, **padded** files.

- `atmos/pad_era5.py` (run as `python pad_era5.py` on PPAN, `:1-6, 151-152`): for each of `msl`, `sp`,
  `tp`, `sf`, `ssrd`, `strd`, `t2m`, `d2m`, `u10`, `v10` (`:36-75`) and years 1993–2021 (`:19`):
  slice lat 0–60°N, lon 260–325°E (`:23, 105`), make `time` the record dimension (`:110`), unpack
  (`ncpdq -U`, "FMS doesn't seem to work with packed data", `:112-114`), flip latitude to south→north
  (`:116-118`), and append the first record of the next year (midnight 1 Jan) to each year
  (`:124-140`) → `ERA5_{var}_{year}_padded.nc`.
- `atmos/era5_lp.sh YEAR` (`:11-13`): liquid precipitation
  `lp = max(tp − sf, 0)` via `cdo -setrtoc,-1e9,0,0 -chname,tp,lp -sub tp sf`.
- `atmos/era5_sphum.py` (years hard-coded 1993–2020, `:60-63`) or `atmos/era5_sphum_cdo.py YEAR`:
  2 m specific humidity from dew point and **surface pressure `sp`** (not `msl`):
  `e_s = 611.2·exp(17.67 (T_d − 273.15)/(T_d − 29.65))`, `r = 0.622 e_s/(sp − e_s)`, `q = r/(1 + r)`
  (`era5_sphum.py:12-36`; `era5_sphum_cdo.py:22-23`). pyOCEAN's N2b ERA5 adapter computes q from `d2m` and **`msl`**
  (`forcing/sources/era5.py:109-132, 289`); the two differ by the ratio `msl/sp`, ≈ 1 over the
  open ocean (class c).
- `atmos/merge_co2_forcing.py`: CO₂ mole fraction for COBALT; not needed for the physical cut.

**Units through `data_table`** (scale column): `strd`, `ssrd` J m⁻² per hour ÷ 3600; `ssrd` split
0.285 / 0.285 / 0.215 / 0.215 into vis-dir / vis-dif / nir-dir / nir-dif (`7.9167e-5 = 0.285/3600`,
`5.9722e-5 = 0.215/3600`, `data_table:14-18`); `lp`, `sf` m of water per hour × 1000/3600
(`0.277777778`, `:19-20`); winds bicubic, the rest bilinear (`:1-6`).

## 5. Rivers: `tools/rivers/write_runoff_glofas.py`

**Command:** `./write_runoff_glofas.py --config runoff_glofas.yaml` (`:4-5, 20-23`).

**Config** (`rivers/runoff_glofas.yaml:1-19`): `grid_mask_file` (`ocean_mask.nc`, variable `mask`),
`grid_mosaic_file` (unused), `hgrid_file`, `ldd_file` (GloFAS local drain direction), `output_dir`,
`start_year`, `end_year`, `glofas_files_pattern` (`GloFAS_river_discharge_{year}.nc`, variable
`dis24`, daily discharge in m³ s⁻¹), `latitude_range`, `longitude_range`.

**Algorithm:**
1. Pour points: GloFAS cells with `ldd == 5` (outlet) that touch the ocean (NaN in `ldd`) in a 3×3
   window, grown iteratively through neighbouring outlets until no change (≤ 20 iterations,
   `:249-266`).
2. Model coast cells: ocean cells with land in one of the four neighbours, excluding the outermost
   rows/columns (domain edges are not coasts; `get_coast_mask`, `:25-41`).
3. Convert discharge to a flux with an **approximate GloFAS cell area**
   `(0.1° · cos φ · 111 km) · (0.1° · 111 km)` (`:92-99`), keep only pour points, and regrid
   **conservatively** (xesmf `conservative`) onto the model T cells (`:101-113`; model cell bounds
   from the supergrid corners `x[::2, ::2]`, area from the four supergrid sub-cells, `:85-90`).
4. NWA12-specific edits in **full-grid indices** (`:115-144`): zero runoff draining to the Pacific
   (Guatemala, El Salvador), on the southern boundary row (`[:, 0:1, :]`, "avoid double counting")
   and in Hudson Bay; Mississippi rescaled to the Belle Chasse gauge
   (`0.5192 Q + 3084.6 m³ s⁻¹`) and moved to three cells near the delta.
5. Move every cell's runoff to its **nearest coastal cell** (xesmf `nearest_s2d` on lon/lat,
   `:146-185`); NWA12 redirects the Susquehanna region to the lower Chesapeake (`:165-171`).
6. Output `glofas_runoff_{year}.nc`: `runoff(time, y, x)` [kg m⁻² s⁻¹], `area`, `lat`, `lon`,
   time in days since 1950-01-01 (`:190-226`).

Remarks for the cut: (a) the hard-coded index edits (steps 4–5) refer to the full 775 × 845 NWA12
grid, so the runoff for a window must be produced on the full grid and then cut (not by running the
script on the cut grid); (b) the flux conversion uses a spherical approximation of the GloFAS area
while the regridding uses ESMF's own cell areas, so Σ runoff·area on the model grid need not equal
Σ discharge exactly [hipótesis, to be measured in Task 8]; (c) the nearest-coast reassignment
conserves the gridded total exactly.

## 6. Grid files and FRE-NCtools: `tools/grid/make_mosaic.sh`, `make_mask.sh`

`ocean_hgrid.nc` and `ocean_topog.nc` are made outside CEFI's tools (ESMG `gridtools` or COSIMA
`regional-mom6`, `grid/README.md:9-52`). From them (`grid/make_mosaic.sh:1-8`, needs `nco` and
`fre/bronx-19`):

```
ncap2 -s 'defdim("ntiles", 1);ntiles[ntiles]=1' ocean_topog.nc -O ocean_topog.nc
make_solo_mosaic --num_tiles 1 --dir . --mosaic_name ocean_mosaic --tile_file ocean_hgrid.nc
make_quick_mosaic --input_mosaic ocean_mosaic.nc --mosaic_name grid_spec --ocean_topog ocean_topog.nc
```

- `make_solo_mosaic` (FRE-NCtools) writes `ocean_mosaic.nc`, which only names the tile file(s) and
  contacts (none for a regional, non-periodic tile).
- `make_quick_mosaic` (FRE-NCtools) uses the **same grid for atmosphere, land and ocean** and writes
  `grid_spec.nc`, `atmos_mosaic`/`land_mosaic` descriptors, the land/ocean masks
  (`land_mask.nc`, `ocean_mask.nc`) from the topography, and the exchange grids
  `atmos_mosaic_tile1Xocean_mosaic_tile1.nc`, `atmos_mosaic_tile1Xland_mosaic_tile1.nc`,
  `land_mosaic_tile1Xocean_mosaic_tile1.nc` (exactly the files present in
  `exps/NWA12.COBALT/INPUT/`). Because the atmosphere grid equals the ocean grid, the exchange grid
  is one cell per ocean cell (`xgrid_nml interp_method = 'second_order'` still applies,
  `input.nml_24hr`). [verificar: FRE-NCtools source is not in `~/pyOCEAN_refs`; the list of outputs is
  from the NWA12 `INPUT/` directory and the tool's documented behaviour, not from its code.]
- `make_mask.sh` runs `check_mask --grid_file ocean_mosaic.nc --ocean_topog ocean_topog.nc --layout
  N,N` for 40×40, 50×50, 60×60 (`:1-3`) to write `mask_table.*` (processor land-masking); not needed
  for a serial or small cut.

**Which tools need FRE-NCtools:** only `make_solo_mosaic`, `make_quick_mosaic` and `check_mask`.
All other tools are Python (+ ESMF via xesmf, HCtFlood) plus NCO/CDO. For the cut on this Mac the
options are (i) install FRE-NCtools (C, autotools; conda-forge has `fre-nctools`
[verificar]) or (ii) write `ocean_mosaic.nc`, `grid_spec.nc`, the masks and the trivial
identity exchange grids in Python, cross-checked against NWA12's own files. Decision for Task 10.

Also in `exps/NWA12.COBALT/INPUT/` (all symlinks into `../../datasets/nwa12_input/`, **broken** in
the local clone, 2026-11-01): `ocean_hgrid.nc`, `ocean_topog.nc`, `topog.nc`, `vgrid_75_2m.nc`,
`glorys_ic…`, `uv/thetao/so/zos/tu/tz_00{1,2,3}.nc`, `ERA5_*_1993_padded.nc`,
`glofas_runoff_1993.nc`. The NWA12 grid tarball being fetched into `~/pyOCEAN_refs/nwa12_grid/`
(`grid/nwa12_grid_75z.tar`, Andrew Ross, 2023-05-11) was still incomplete when this was written.

## 7. Summary: one command line per input (cut workflow sketch for Task 10)

| input | tool / command | needs |
|---|---|---|
| `ocean_hgrid.nc`, `ocean_topog.nc` (cut) | slice the full NWA12 files with even supergrid offsets (`mom6_grid.md` §7) | xarray |
| `ocean_mosaic.nc`, `grid_spec.nc`, masks, exchange grids | `make_mosaic.sh` | FRE-NCtools, NCO |
| `uv/thetao/so/zos_00n.nc` | `write_glorys_boundary.py --config glorys_obc.yaml` | xesmf, NCO, flooded GLORYS (CDO) |
| `tu/tz_00n.nc` | `write_tpxo_boundary.py --config tpxo_obc.yaml` | xesmf, HCtFlood, TPXO9 v1 |
| `glorys_ic_YYYY-MM-DD.nc` | `write_glorys_initial.py --config_file glorys_ic.yaml` | xesmf, HCtFlood |
| `ERA5_*_YYYY_padded.nc` | `pad_era5.py`, `era5_lp.sh YEAR`, `era5_sphum_cdo.py YEAR` | NCO, CDO |
| `glofas_runoff_YYYY.nc` | `write_runoff_glofas.py --config runoff_glofas.yaml` (full grid, then cut) | xesmf, GloFAS `dis24` + `ldd` |
