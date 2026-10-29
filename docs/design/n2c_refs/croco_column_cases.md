# CROCO single-column (SINGLE_COLUMN) test cases and KPP — reference read

Source: `/Users/estebancruz/pyOCEAN_refs/croco/` (read only). All `file:line` below are relative to `OCEAN/` unless stated (`TC/` = `TEST_CASES/`). Reference implementation, not truth; items from memory are marked **[verificar]**.

## 0. Common to all seven cases

| Item | Value | file:line |
|---|---|---|
| Selector | `#define SINGLE_COLUMN` and exactly one case key; only `KATO_PHILIPS` is defined in the shipped file, the other six are `#undef` | cppdefs.h:758-778 |
| Options in the block | `UV_ADV`, `NEW_S_COORD`, `UV_COR`, `SOLVE3D`, `#undef LMD_MIXING`, `#define GLS_MIXING`, `ANA_GRID/INITIAL/SMFLUX/SRFLUX/STFLUX/BTFLUX`, `EW_PERIODIC`, `NS_PERIODIC`, `#undef MPI/OPENMP` | cppdefs.h:779-792 |
| GLS variant | not written in the SINGLE_COLUMN block; defaults to `GLS_KEPSILON` + `CANUTO_A` (also `GLS_MIXING_3D` not set) | cppdefs_dev.h:541-561 |
| Tracers | `TEMPERATURE` only (no `SALINITY` in the block), linear EOS (no `NONLIN_EOS`): `rho = R0 - Tcoef (T - T0)` (+ rho0 in the dynamics) | cppdefs_dev.h:127; rho_eos.F:360-362 |
| EOS constants (all .in files) | rho0 = 1024 (EKBBL: 1.22), R0 = 24, T0 = 16 degC, Tcoef = 0.2 kg m-3 K-1 (alpha = Tcoef/rho0 = 1.95e-4 K-1), Scoef = 0 | TC/croco.in.* lines 43-47 (EKBBL rho0 line 44 `1.22d0`) |
| Horizontal grid | 5 x 5 interior (LLm0=MMm0=5), Length_XI = Length_ETA = 10 km (dx = dy = 2 km), doubly periodic, flat bottom `h = depth`; so a true 1-D column (all columns identical) | param.h:78-93; ana_grid.F:297-300, 536-545 |
| Vertical grid | uniform sigma: `THETA_S = 0, THETA_B = 0, Hc = 1e16` (so z_w uniformly spaced, dz = depth/N) | TC/croco.in.* line 5-6 |
| Time stepping | `NTIMES dt NDTFAST NINFO`; dt = 30 s in every case (NDTFAST = 60 is irrelevant, no barotropic dynamics in a periodic flat box) | TC/croco.in.* line 3-4 |
| Mixing background (.in) | `Akv_bak = 1e-5`, `Akt_bak = 1e-6` m2/s (NONROTBBL: `Akt_bak = 1e-5`). Read at read_inp.F:2471. GLS has its own hard-wired backgrounds `nuwm = 1e-5`, `nuws = 1e-6` (gls_mixing.F:~150-153, `# else` branch) | TC/croco.in.* line 59-60 |
| Bottom drag (.in) | `RDRG = RDRG2 = 0`, `Zob` per case (below), `Cdb_min = 0`, `Cdb_max = 0` (1e7 for FORCED_*) | TC/croco.in.* line 62-63 |
| GLS numerics | `tke_min = 1e-10`, `eps_min = 1e-12`, `Zosmin = 1e-2 m` (surface roughness floor), `Zobmin = 1e-4 m`, Charnock `chk = 1400/g`, `galp = 0.53` | gls_mixing.F:144-163 |
| GLS BCs | surface: z0_s = max(Zosmin, chk u*^2), length scale `vonKar (0.5 Hz_N + z0_s)`; bottom: z0_b = max(Zob, Zobmin) | gls_mixing.F:694-717 |
| Outputs | `scm_his.nc` (zeta, ubar, vbar, u, v, T every NWRT = 120 steps = 1 h; Akv/Akt/Aks switched on only in DIURNALCYCLE) and `scm_avg.nc` (1-h averages: u, v, T, Omega, W, Akv, Akt, HBL, HBBL, bottom/surface stresses, plus GLS `Akk Akp TKE GLS Lscale`) | TC/croco.in.* lines 23-40 |
| Filename quirk | read_inp.F:127 points to `TEST_CASES/croco.in.KATO_PHILLIPS` (double L) but the file is `croco.in.KATO_PHILIPS`; rename or symlink when building | read_inp.F:127 |
| Compiled run in `croco_runs/` | **None.** Existing: `grav_adj`, `grav_adj_uniform`, `thacker`, `tidal_flat`, `tidal_flat3d`, `tidal_flat3d_matched` | `/Users/estebancruz/pyOCEAN_refs/croco_runs/` |
| Benchmark harness entries | the seven cases are registered (`BENCH/config.d/cases/single-column.jsonc`, `check_outputs: scm_his.nc`); its titles list EKBBL as `5x5x40 : 23040` but the .in has NTIMES = 2880 (title is stale) | BENCH/config.d/cases/single-column.jsonc:1-40 |
| Analytical or LES reference | **None in the source tree** (no plot script, no README mention; grep for Kato/Phillips/Willis/Deardorff finds only the cppdefs). The references below are the standard literature ones, from memory **[verificar]** | n/a |

## 1. KATO_PHILIPS — entrainment under constant wind stress

| Item | Value | file:line |
|---|---|---|
| Grid | 5 x 5 x **N = 100**, depth 50 m, dz = 0.5 m | param.h:81-82; ana_grid.F:119-122 |
| Coriolis | f = 0, beta = 0 | ana_grid.F:119-122 |
| Initial T | `T = 16 + (0.01)^2/(g Tcoef/rho0) z` (z<0), i.e. constant N = 0.01 s-1 (N^2 = 1e-4 s-2) with T = 16 degC at z = 0; at the bottom (-50 m) T = 16 - 0.01^2/(g*1.953e-4)*50 = about 13.4 degC. No salinity. u = v = 0 | ana_initial.F:590-599 |
| Surface wind stress | `sustr = (0.01)^2 = 1e-4 m2 s-2`, i.e. u* = 0.01 m/s, `svstr = 0` (default) | analytical.F:378-383 |
| Surface heat/buoyancy flux | `stflx = 0`, `srflx = 0` | analytical.F:879-881 (else branch), analytical.F:545-551 |
| Bottom flux | `btflx(T) = -1e-5 (0.01)^2/(g Tcoef/rho0)` = -1e-5 * 1e-4/(9.81*1.953e-4) = about -5.2e-7 degC m/s (small but non-zero; match it or set both to 0) | analytical.F:133-134 |
| Bottom drag | `Zob = 0`, RDRG = 0, so only GLS floor `Zobmin = 1e-4`; bottom stress effectively 0 by `Cdb_min = Cdb_max = 0` | TC/croco.in.KATO_PHILIPS:62-63 |
| EOS | linear, rho0 = 1024, R0 = 24, T0 = 16, Tcoef = 0.2 | .in:43-47 |
| Closure | GLS k-epsilon, Canuto A (`GLS_MIXING`, `LMD_MIXING` undefined) | cppdefs.h:779-785 |
| dt, run | dt = 30 s, NTIMES = 3600, so **30 h** | .in:3-4 |
| Outputs | scm_his/avg hourly | .in:23-40 |
| Comparison | no reference in source. Kato & Phillips (1969) / Price (1979): h(t) = 1.05 u* (t/N)^{1/2} up to a constant (Price et al. 1978 laboratory fit, about 1.05); with u* = 0.01, N = 0.01: h about 1.05 * 0.01 * sqrt(t/0.01) = 1.05 * sqrt(t)/10 m, so h(30 h = 108000 s) about 34 m **[verificar constants]**. Known: k-epsilon models with Canuto A or Kantha-Clayson give h within about 10-20 % of the K-P law; compare with GOTM tests **[verificar]** | n/a |

## 2. WILLIS_DEARDORFF — erosion of linear stratification by surface cooling (convection)

| Item | Value | file:line |
|---|---|---|
| Grid | 5 x 5 x **N = 50**, depth 50 m, dz = 1 m | param.h:83-85; ana_grid.F:119-122 |
| Coriolis | f = 0 | ana_grid.F:119-122 |
| Initial T | `T = 0.1 z + 22` (z<0), i.e. dT/dz = 0.1 K/m, N^2 = g alpha dT/dz = 9.81*1.953e-4*0.1 = 1.92e-4 s-2 (N = 0.0138 s-1); T(surface) = 22, T(-50 m) = 17 degC. u = v = 0 | ana_initial.F:568-577 |
| Wind stress | none (`sustr` is not defined for this case; defaults to 0) | analytical.F:378-390 (no branch) |
| Surface heat flux | `stflx(T) = -100/(rho0 Cp)` with Cp = 3985, rho0 = 1024, so Q = -100 W m-2 (cooling), kinematic -2.45e-5 degC m/s; buoyancy flux B0 = g alpha |stflx| = 4.7e-8 m2 s-3 | analytical.F:876-877; scalars.h:640 |
| Shortwave | `srflx = 0` | analytical.F:545-551 |
| Bottom flux | `btflx(T) = -1e-5 * 0.01 = -1e-7` degC m/s (small non-zero, as KATO) | analytical.F:135-136 |
| Bottom drag | Zob = 0 (GLS floor 1e-4) | .in:62-63 |
| Closure | GLS k-epsilon Canuto A | cppdefs.h:779-785 |
| dt, run | dt = 30 s, NTIMES = 8640, so **72 h** | TC/croco.in.WILLIS_DEARDORFF:3-4 |
| Comparison | no reference in source. Willis & Deardorff (1974, laboratory convection tank), Deardorff-type LES; the penetrative-convection mixed-layer-depth law from the energy budget (zero-order, Kraus-Turner, with entrainment ratio about 0.2): h^2 = (2(1+2A)... ) B0 t / N^2... **[verificar]**; practical check: mixed-layer heat content loss equals the integral of Q (conservation), h(t) from the equal-area construction h^2 = 2 (1+... ) B0 t/N^2 with entrainment constant A about 0.2 **[verificar]** | n/a |

## 3. DIURNAL_CYCLE — daily shortwave plus constant cooling on weak stratification

| Item | Value | file:line |
|---|---|---|
| Grid | 5 x 5 x **N = 150**, depth 150 m, dz = 1 m | param.h:90-91; ana_grid.F:122-125 |
| Coriolis | f = 0 | ana_grid.F:122-125 |
| Initial T | `T = 0.01 z + 22` (dT/dz = 0.01 K/m; N^2 = 1.92e-5 s-2); T(0) = 22, T(-150) = 20.5 degC | ana_initial.F:579-588 |
| Wind stress | none | analytical.F |
| Shortwave | `srflx = 264.91/(rho0 Cp) max(cos(2 pi (t/86400 - 0.5)), 0)`: half-rectified cosine, peak 264.91 W m-2 at local noon (t = 12 h within each day), zero at night; daily mean about 84.3 W/m2 | analytical.F:553-557 (else branch of the `INNERSHELF` test, line 548-558) |
| Non-solar flux | `stflx(T) = srflx - 75/(rho0 Cp)`: total surface flux includes srflx (CROCO convention) minus a constant 75 W m-2 loss; daily mean net = 84.3 - 75 = +9.3 W m-2 | analytical.F:878-879 |
| Shortwave penetration | handled in `step3d_t.F` (below, section 5) with Jerlov type 1 (`lmd_Jwt = 1`) since `GLS_MIXING` is defined | step3d_t.F:578-586; lmd_swfrac.F:54, 137 |
| Bottom flux | `btflx = -1e-7` degC m/s | analytical.F:135-136 |
| Closure | GLS k-epsilon Canuto A | cppdefs.h |
| dt, run | dt = 30 s, NTIMES = 23040, so **8 days** | TC/croco.in.DIURNALCYCLE:3-4 |
| Outputs | as common, plus `Akv Akt Aks` in the history file | .in:31-32 |
| Comparison | no reference in source. Natural check: diurnal-cycle mixed layer from GOTM/KPP intercomparisons (Large et al. 1994 OWS Papa, McWilliams et al. 2009 on the need of Ric = 0.15 with diurnal shortwave) **[verificar]**; internal consistency: conservation of heat content, nocturnal convective mixing depth | n/a |

## 4. Forced bottom-boundary-layer cases (no stratification, T = T0 = 16 uniform)

For all four: initial T = T0 uniform (default branch), u = v = 0, no surface heat flux, `stflx = srflx = 0`, bottom flux 0, GLS k-epsilon Canuto A, linear EOS (irrelevant: no stratification), dt = 30 s. The forcing is a body force added to the pressure-gradient term in `prsgrd.F`: `rv += ... on_v ( f*UG | -gdzetady [cos(omega t)] ) + ...`. All act on the **v** component (alongshore), u is driven only by rotation. Surface boundary is free slip (no stress) unless noted.

| Item | FORCED_EKBBL | FORCED_DBLEEK | FORCED_NONROTBBL | FORCED_OSCNONROTBBL |
|---|---|---|---|---|
| Physical setup | Ekman bottom boundary layer under a steady geostrophic flow (atmosphere-like parameters) | Ekman layers at surface and bottom of a shallow water column (double Ekman) | Non-rotating bottom boundary layer under a steady pressure gradient (Prandtl/log layer, steady turbulent channel-type flow) | Oscillating non-rotating BBL (tidal Stokes-type boundary layer) |
| Grid | 5x5x**40**, depth **1500 m** (dz = 37.5 m) | 5x5x**25**, depth **25 m** (dz = 1 m) | 5x5x**50**, depth **5 m** (dz = 0.1 m) | 5x5x**50**, depth **5 m** (dz = 0.1 m) |
| grid refs | param.h:86-87; ana_grid.F:126-129 | param.h:88-89; ana_grid.F:130-133 | param.h:83-85; ana_grid.F:134-137 | same as NONROTBBL |
| f0 | **1e-4 s-1** | **1e-4 s-1** | 0 | 0 |
| rho0 | **1.22** (air) | 1024 | 1024 | 1024 |
| Forcing | along-v geostrophic gradient `f UG` with **UG = 10 m/s**, i.e. dp/dy/rho0 = -f UG = -1e-3 m s-2 (prsgrd.F:67-69, 457-458) | `f UG` with **UG = 0.0015 m/s** plus a surface wind stress `sustr = -(0.0025)^2 = -6.25e-6 m2 s-2` (u* = 0.0025 m/s, westward) (prsgrd.F:70-72; analytical.F:384-389) | uniform along-v pressure gradient `-gdzetady`, **g dzeta/dy = 9.81*2e-6 = 1.962e-5 m s-2** (prsgrd.F:73-76, 461-462) | `-gdzetady cos(omega t)` with the same amplitude 1.962e-5 m s-2 and **omega = 2 pi/(12.4 h)** (M2 period, T = 12.4 h) (prsgrd.F:73-76, 459-460) |
| Bottom roughness Zob (.in) | **0.1 m** | **0.05 m** | **0.01 m** | **0.01 m** |
| Cdb_min / Cdb_max | 0 / 1e7 | 0 / 1e7 | 0 / 1e7 | 0 / 1e7 |
| Akv_bak / Akt_bak | 1e-5 / 1e-6 | 1e-5 / 1e-6 | 1e-5 / **1e-5** | 1e-5 / 1e-6 |
| NTIMES x dt | 2880 x 30 = **24 h** | 2880 x 30 = **24 h** | 3600 x 30 = **30 h** | 7200 x 30 = **60 h** (about 4.8 tidal periods) |
| .in file | TC/croco.in.FORCED_EKBBL (rho0 at line 44, Zob line 63) | TC/croco.in.FORCED_DBLEEK | TC/croco.in.FORCED_NONROTBBL | TC/croco.in.FORCED_OSCNONROTBBL |
| Expected analytical/lab check (no reference in the source tree) | Steady-state Ekman spiral: with constant K, Ekman depth sqrt(2K/f); with the log layer, u* from the Zob and the geostrophic drag law (Zilitinkevich, Csanady constants A, B) **[verificar]**; this is the standard 1-D BBL test of the CROCO documentation / Warner et al. 2005 **[verificar]** | Two-layer Ekman solution with K constant, total transport = wind stress / f (Ekman transport) plus geostrophic flow UG; check the transport and the cross-layer stress balance | Steady turbulent channel flow: u(z) = (u*/kappa) ln(z/z0) with u* = sqrt(g dzeta/dy * H) (full-depth momentum balance), Prandtl layer; kappa = 0.41, check log-law slope and u* | Stokes second-problem boundary layer, phase lag and amplitude of the near-bed velocity vs the depth-mean oscillating flow; log-layer plus oscillating forcing (Jensen et al. 1989 laboratory/DNS in the literature) **[verificar]** |
| Notes | the unusual rho0 = 1.22 and depth 1500 m mimic an atmospheric BL (Ekman layer in the atmosphere); in CROCO this is the same code with air-like constants | the surface stress in the DBLEEK case is the only forcing case with wind | no rotation so the solution is 1-D in v only | requires the time-dependent term to be resolved: dt = 30 s gives 1490 steps per period |

## 5. CROCO KPP (not used in the single-column cases, since `LMD_MIXING` is undefined there)

Enabled in the regional presets with `LMD_MIXING` + `LMD_SKPP`, `LMD_BKPP`, `LMD_RIMIX`, `LMD_CONVEC`, `LMD_NONLOCAL` (cppdefs.h:255-259, 742-745). `LMD_DDMIX` (double diffusion) is undefined by default; `LMD_LANGMUIR` undefined (cppdefs.h:260-261).

| Item | Detail | file:line |
|---|---|---|
| Version of the surface scheme | Default: **if-less KPP 2005** (`LMD_SKPP2005` is defined automatically when `LMD_SKPP` is defined); the original Large et al. (1994) is `lmd_skpp1994.F`, used only if `LMD_SKPP2005` is undefined | cppdefs_dev.h:753-755; lmd_skpp.F:11-19; lmd_skpp2005.F:19-45 |
| Algorithm | hbl = first depth where the bulk Richardson function `Cr(z) = int Kern (du^2/dz'^2 - N^2/Ric - C_Ek f^2) dz' + Vt^2/(z - zeta)` crosses zero (Shchepetkin 2005 "If-less KPP"; Large et al. 1994), Kern = zscale/(zscale + epssfc hbl) | lmd_skpp2005.F:19-30, 299, 319-325 |
| Constants (2005) | Cv = 1.8, **Ric = 0.45** (but 0.15 if `DIURNAL_SRFLUX`, `ROBUST_DIURNAL_SRFLUX`, `ANA_DIURNAL_SW`, `BULK_FLUX` or `OA_COUPLING`), betaT = -0.2, epssfc = 0.1, Cstar = 10, nu0c = 0.1 m2/s, C_Ek = 258, zeta_m = -0.2, a_m = 1.257, ... Cg = Cstar kappa (c_s kappa epssfc)^(1/3); Vtc = Cv sqrt(-betaT/(c_s epssfc)) / (Ric kappa^2) | lmd_skpp2005.F:95-117, 159-160 |
| Constants (1994 variant) | Ric = 0.3 | lmd_skpp1994.F:72 |
| Nonlocal term | `ghats = Cg sigma (1 - sigma)^2` only for destabilising surface buoyancy flux (Bfsfc < 0); applied in the tracer equation as `-dt ghats (stflx - srflx)` for T and `-dt ghats stflx_S` for S (momentum has no nonlocal term) | lmd_skpp2005.F:488-492, 512-513; step3d_t.F:584-586, 598-600 |
| Convective mixing | `nu0c = 0.1 m2/s` added to Kv, Kt, Ks inside the boundary layer when stratification is unstable (and below hbl via `LMD_CONVEC` in lmd_vmix.F) | lmd_skpp2005.F:500-527; lmd_vmix.F:387-388 |
| Bottom boundary layer | `LMD_BKPP` uses **bkpp1994** by default (the 2005 bottom version has "unresolved problems at depth", so `LMD_BKPP2005` is #undef); profile `Kz = vonKar ustarb z` when hbl reaches the bottom | cppdefs_dev.h:757-760; lmd_bkpp1994.F; lmd_skpp2005.F:410-422 |
| Interior mixing (`lmd_vmix.F`) | Large et al. 1994 shear mixing (Ri0 = 0.7, `nu0m = nu0s = 50e-4 m2/s`, Ri-dependent cubic ratio `(1 - (Ri/Ri0)^2)^3`), internal wave background `lmd_nuwm = 1e-4`, `lmd_nuws = 0.1e-4 m2/s` (these are KPP's own background values, **not** the `Akv_bak`/`Akt_bak` of the .in file, which are applied later as `max(Akv, Akv_bak)` at lmd_vmix.F:~661-663), convective `nu0c = 0.1`, double diffusion off by default (`lmd_Rrho0 = 1.9`, `lmd_nuf = 10e-4`) | lmd_vmix.F:108-137, 344-388, 661-663 |
| Surface buoyancy flux | `Bfsfc` built from `g (alpha (stflx_T - srflx) ...)`: the non-solar part of the heat flux plus the penetrative contribution (srflx enters through the shortwave fraction at the depth tested), salt flux with beta | lmd_skpp2005.F:180-186 (T); lmd_skpp1994.F:188 |
| Velocity scale | `ws` from the (Large et al. 1994) Monin-Obukhov similarity functions, the 2005 version uses the if-less formulation (`lmd_wscale_wm_and_ws.h`, `lmd_wscale_ws_only.h`) | lmd_wscale_wm_and_ws.h; lmd_wscale_ws_only.h |
| Shortwave penetration | `lmd_swfrac.F` computes the fraction `swdk(k)` of srflx still present at the interface k with the **two-band Paulson & Simpson (1977)** formula for Jerlov water types I (default, `lmd_Jwt = 1`) to V; coefficients `mu1 = 0.35,0.6,1.0,1.5,1.4 m`, `mu2 = 23,20,17,14,7.9 m`, `r1 = 0.58,0.62,0.67,0.77,0.78` (r2 = 1 - r1) | lmd_swfrac.F:16-34, 54, 137; lmd_kpp.h:13-23 |
| How T receives fluxes | In `step3d_t.F` the implicit vertical diffusion is solved with flux array `FC`: top `FC(N) = dt stflx` (stflx includes srflx), bottom `FC(0) = -dt btflx`, interior `FC(k) = dt srflx swdk(k) - dt ghats (stflx - srflx)` (the second only with `LMD_NONLOCAL`). This block is active when `LMD_SKPP`, `LMD_BKPP` **or `GLS_MIXING`** is defined, so GLS columns also get Jerlov-I penetration | step3d_t.F:563-587; lmd_swfrac call at step3d_t.F:453 |
| Salt flux | `stflx(S)` is a kinematic freshwater-type flux scaled by surface salinity in `step3d_t` | analytical.F:886-892 |
| Default tracers' background | `Akt_bak(itrc)` from the .in file, applied as a minimum | read_inp.F:2471 |
| HBL diagnostics | `hbls` saved with a one-step lag (`hbls(:,:,3-nstp)`), output `HBL`, `HBBL` | lmd_skpp2005.F:544-597 |

## 6. What building one case would take (not done)

No SINGLE_COLUMN binary exists in `croco_runs/`. To build, e.g., KATO_PHILIPS:

1. Create a run directory like those in `croco_runs/` (copy the structure of `thacker/`: `create_config.bash` output, `Compile/` with `jobcomp`, `param.h`, `cppdefs.h`).
2. In `cppdefs.h` set `#define SINGLE_COLUMN`, keep `#define KATO_PHILIPS` (or flip to the other key; only one at a time). `param.h` picks N from the key (param.h:78-93). Build with the standard `jobcomp` (serial, no MPI/OpenMP: the block `#undef MPI/OPENMP`). Estimated compile: a few minutes, NetCDF + gfortran as for the existing runs.
3. Work around the filename typo (read_inp.F:127: `croco.in.KATO_PHILLIPS`): symlink `TEST_CASES/croco.in.KATO_PHILLIPS -> croco.in.KATO_PHILIPS`, or pass the .in file explicitly on the command line (`./croco croco.in`).
4. Run: 3600 steps of 30 s, a few seconds of CPU (5x5x100 columns). Output `scm_his.nc` / `scm_avg.nc` with u, v, T, Akv, Akt, TKE, GLS, Lscale, HBL.
5. For a comparison with the pyOCEAN column: use the 1-h averages of the central column, same N, dz, dt = 30 s, same EOS (linear, Tcoef = 0.2), same background (`Akv_bak = 1e-5`, `Akt_bak = 1e-6`), `Zosmin = 1e-2`, `chk = 1400/g`.
6. Each other case = rebuild with the other key (7 compilations; N and depth change in `param.h`). Reuse the same `Compile/` tree.
7. LMD/KPP in a single column is **not** configured by this block (`LMD_MIXING` undefined). To run KPP: `#undef GLS_MIXING`, `#define LMD_MIXING` (+ `LMD_SKPP`, `LMD_BKPP`, `LMD_RIMIX`, `LMD_CONVEC`, `LMD_NONLOCAL`, as in cppdefs.h:255-259), and for DIURNAL_CYCLE `ANA_DIURNAL_SW` or equivalent so that `Ric = 0.15` is selected (lmd_skpp2005.F:98-103).

## 7. Notes for the pyOCEAN column validation

- The CROCO `stflx` convention is net flux including shortwave; `srflx` is subtracted for the nonlocal and `Bfsfc` terms and re-added through `swdk` (step3d_t.F:578-587, lmd_skpp2005.F:180).
- Kato-Phillips and Willis-Deardorff have f = 0 and no solar radiation, so they test only the turbulence closure (no Ekman damping, no Ric adjustment).
- The small non-zero bottom heat flux in KATO/WILLIS/DIURNAL (`btflx` about 1e-7 degC m/s, analytical.F:133-136) is a CROCO detail that should be matched or set to zero in both models before comparing T budgets.
- Several GLS values in CROCO to match: `Zosmin`, Charnock `1400/g`, `tke_min`, `eps_min`, `galp = 0.53`, the hard-wired backgrounds (1e-5, 1e-6).
- Items marked **[verificar]**: analytical/lab references (K-P, Willis-Deardorff, Ekman/Prandtl layers, Jensen et al. 1989, McWilliams et al. 2009), because the CROCO tree ships no analytical comparison for these cases.
