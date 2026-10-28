# How CROCO composes its time step (for pyOCEAN N1)

Source: `~/pyOCEAN_refs/croco/OCEAN/`, commit ab63b58 ("Tag 2.1.3"). All paths below are relative to `OCEAN/`. CROCO 2.x has no `main3d.F`, and `prestep3D_thread`, `step2d_thread`, `step3D_uv_thread` and `step3D_t_thread` are not separate files: all of them live in `step.F`. There is no `checkdefs.F` either. Its role is split between a generated `check_switches1.F` and the hand-written `check_switches2.F`.

## 1. Call order of one baroclinic step

### Outer loop: a state machine driven by `iif`
`main.F:950` loops over `iicroot = ntstart..ntimes+1`. Each baroclinic step calls `step()` **nfast+3 times** (`main.F:966-977`, `do iifroot = 0,nfast+2`). `step()` (`step.F:17-101`) increments `iif` (`step.F:48`) and dispatches on it:

| `iif` | phase | lines |
|---|---|---|
| 0 | `prestep3D_thread` | `step.F:49-53` |
| 1..nfast | `step2d_thread`: **one** barotropic step per call | `step.F:56-60` |
| nfast+1 | `step3D_uv_thread` | `step.F:63-67` |
| nfast+2 | `step3D_t_thread`, then `iif=-1` and `nbstep3d++` | `step.F:70-77` |

Each phase is wrapped in `C$OMP PARALLEL`. The phases are re-entrant this way so that AGRIF can interleave parent and child grids (`main.F:972`, `Agrif_Step(step)`).

### Phase 0: `prestep3D_thread` (`step.F:108-616`)
1. Time and time levels: `time=time_start+dt*(iic-ntstart)` (`:180`), `nstp=1+mod(iic-ntstart,2)`, `nrhs=nstp`, `nnew=3` (`:188-190`). The predictor writes into level 3, which holds n+½.
2. Read boundary and forcing data at file-record changes (`synchro_flag`): `get_tclima`, `get_uclima`, `get_ssh`, `get_bry`, `get_vbc` (`:217-245`).
3. Tile loop A (`:257-322`): time interpolation of climatology, ssh and boundary data (`set_tclima`/`set_uclima`/`set_ssh`/`set_bry`), then **`rho_eos`** (`:310`), **`set_HUV`** (`:311`, face fluxes Huon/Hvom from u(n)), and `diag` (`:315`). `diag` computes energies and sets `may_day_flag` on blow-up (`diag.F:365`).
4. Tile loop B (`:324-342`): **`set_vbc`** (surface and bottom stress and fluxes, `:325`), `bblm`, `clm_tides` (`:339`).
5. `bulk_flux` (+ ABL) (`:364-371`). Waves (WKB/MRL_WCI) come next (`:411-472`).
6. Tile loop C (`:474-486`): **vertical mixing**, `ana_vmix` / **`lmd_vmix` (KPP)** / `bvf_mix` (`:475-481`), then **`omega`** (`:482`) and `hvisc_coef`. KPP is computed once per step from the state at time n.
7. Tile loop D (`:489-505`): **`prsgrd`** (`:490`), **`rhs3d`** (`:491`), then **`pre_step3d`** (`:499`).
   - `rhs3d` builds `ru,rv` from Coriolis, curvilinear terms and advection on top of the `prsgrd` output. At the end it **vertically integrates them into `rufrc,rvfrc`** and adds `(sustr−bustr)·area` (`rhs3d.F:1869-1900`).
   - `pre_step3d` is the **LF-AM3 predictor** with γ=1/6: `cff=(1-γ)dt`, `cff1=½+γ`, `cff2=½−γ`, and Euler with dt/2 on the first step (`pre_step3d.F:183-192`). It produces `u(…,nnew=3)` at n+½ from Hz-weighted `u(n)` and `u(n−1)` (`pre_step3d.F:654-674`). It also stores `Hz·u(n)` in `u(…,indx=3−nstp)` as the accumulator for the corrector (`pre_step3d.F:624, 676`). Tracers are handled the same way (`pre_step3d.F:387-444`). The predictor's implicit vertical part (`tridiag_pred.h`) runs only with `VADV_ADAPT_IMP` (`pre_step3d.F:575-584`). `pre_step3d` applies `t3dbc/u3dbc/v3dbc` to the predictor fields (`pre_step3d.F:1016-1032`). It saves `Akv_old/Akt_old` for GLS (`:1085-1100`). It sets `zeta(knew)=Zt_avg1` to start the fast mode (`:1116-1125`).
8. Tile loop E (`:508-569`): **`uv3dmix`** for horizontal viscosity (`:510`) and `uv3dmix_spg` for the sponge (`:513`). `uv3dmix_S.F` uses `u(nstp)` (time n, lagged for stability) and adds `dt·visc` to the corrector accumulator `u(indx)`. It **also adds the term to `rufrc`** (`uv3dmix_S.F:241-252`), so the barotropic mode feels it. This loop also runs `set_diags_vrt/ek/pv` (`:520-528`) and `set_avg` with the diagnostics averages (`:530-568`).
9. `nrhs=3; nnew=3−nstp` (`:592-593`): from here on the right-hand sides use the n+½ fields. Then **`call output`** (`:612`) writes history, restart and averages. These hold time-n data, e.g. `wrt_rst` writes `u(nstp)` (`wrt_rst.F:345`).

### Phases 1..nfast: `step2d_thread` → `step2d` (`step.F:628-865`, `step2d.F`)
- Fast time levels cycle 1..4 through `kstp/knew` (`step.F:717-719`). Tiles are visited forward or backward depending on the parity of `knew` (`step.F:800-831`), which is a cache trick.
- `step2d.F` uses a **generalized forward–backward AB3 (ζ) / AM4 (ū) scheme** (Shchepetkin & McWilliams 2005, 2009) with β=0.281105 and γ, ε as tuned in the source (`step2d.F:60-95`). The coefficients ramp up from Euler FB on the first fast step to AB2-AM3 on the second (`step2d.F:206-232`).
- **Coupling on the first fast step only** (`FIRST_2D_STEP`, `step2d.F:1277-1350`). With `PRED_COUPLED_MODE` (`#define`d in `step2d.F:15`), `rufrc` becomes `rufrc − rubar` (3-D minus 2-D r.h.s.). This is AB3-extrapolated to n+½ with `rufrc_bak` (weights 1.5+β, −0.5−2β, β; `:1279-1307`) and then **held fixed for all fast steps**. A correction moves the first step's surface-pressure term from ζ(kstp) to ζ_new (`:1320-1360`). On each fast step ū is updated with `rubar+rufrc` (`step2d.F:1408-1411`).
- Calls inside step2d: `set_DUV_bc_tile` (`:337`), **`wetdry_tile`** (`:560`), **`zetabc_tile`** (`:617`), **M2 nudging** with `M2nudgcof` (`:1520-1552`), **`u2dbc_tile/v2dbc_tile`** (`:1611-1612`) and `obc_flux_tile` (`:1615`).
- **Time filtering** (`step2d.F:629-650, 1402-1437`): `Zt_avg1 = Σ weight(1,iif)·ζ`. `DU_avg1(nnew)` gets the primary-weighted transport, with weight ½·weight(1,iif) per half of the FB pair. `DU_avg2 = Σ weight(2,iif)·DUon` uses the secondary weights. The weights come from `set_weights.F`. The default is `M2FILTER_POWER` (`cppdefs_dev.h:107-116`): shape ξ^p(1−ξ^q)−rξ with p=2, q=4, r=0.25, and its centroid is iterated exactly onto `ndtfast` (`set_weights.F:58-80`). So **nfast > ndtfast**: the fast mode runs past t+dt, up to 2·ndtfast (`read_inp.F:424-430` checks `NWEIGHT ≥ 2·ndtfast−1`). The secondary weights are cumulative sums of the primary ones (`set_weights.F:168-182`). `M2FILTER_NONE` gives `nfast=ndtfast` and weight 1 on the last step (`set_weights.F:102-105`).

### Phase nfast+1: `step3D_uv_thread` (`step.F:874-1002`)
1. **`set_depth`** (`:939-941`): new z_r, z_w and Hz from the filtered `Zt_avg1`. `Hz_bak` keeps the old Hz (`set_depth.F:203-211`).
2. **`set_HUV2`** (`:945-947`): the face fluxes are corrected so that their vertical sum equals `DU_avg2` (`set_depth.F:619-670`).
3. **`omega`, `rho_eos`** (`:950-956`), now with n+½ fields (`nrhs=3`).
4. **`prsgrd`, `rhs3d`, `step3d_uv1`** (`:960-978`). The full r.h.s. is recomputed at n+½. `step3d_uv1` adds `dt·ru` to the Hz·u accumulator (`step3d_uv1.F:95-121`). Together with the predictor's use of time n this makes the scheme **LF-AM3 predictor + AM3-type corrector**. The second `rhs3d` call overwrites `rufrc`, but that value is no longer used in this step.
5. **`step3d_uv2`** (`:981-986`) handles:
   - implicit vertical viscosity, a tridiagonal solve with `FC=−dt·Akv/Δz` (`step3d_uv2.F:200-330`);
   - **baroclinic–barotropic coupling**: the depth mean of the new u is replaced by `DU_avg1(nnew)/(D·on_u)` (`:323`, `:930-961`);
   - **`wetdry_tile`** (`:875`) and **`u3dbc/v3dbc`** (`:883-884`);
   - the new `Huon = ¾·Huon + ⅛·Hz(u^n+u^{n+1})`, corrected to `DU_avg2` (`:985-1015`);
   - **M3 nudging** (`:1335-1390`).

   `hdiff_coef` comes after it.

### Phase nfast+2: `step3D_t_thread` (`step.F:1012-1252`)
1. **`omega`** (`:1091`) from the coupled Huon, so tracer advection uses barotropically consistent fluxes.
2. **`gls_mixing`** (`:1093-1095`). GLS is placed **after** the momentum corrector, i.e. it uses the new u, v, unlike KPP in phase 0.
3. **`step3d_t`** (`:1100`) handles:
   - horizontal advection: `t(nnew) = Hz_bak·t(nstp) − dt·pm·pn·div(FX,FE)` (`step3d_t.F:319-321`);
   - vertical advection (`:500-550`);
   - surface, bottom and shortwave forcing (`:555-635`, `lmd_swfrac` `:453`);
   - **implicit vertical diffusion** (`t3dmix_tridiagonal.h`, `:639-735`);
   - **`t3dbc`** (`:796`), biology (`:804`) and **T nudging** with `Tnudgcof` (`:917-943`).

   Sediment and MUSTANG are called around it.
4. **`t3dmix`** and **`t3dmix_spg`** (`:1155-1163`): horizontal diffusion is a **separate, later** update to `t(nnew)`, using the n+½ gradients (`nrhs`) and dividing by the new Hz (`t3dmix_S.F:207-242`).
5. Then morphodynamics, floats and stations. `iic=iic+1` (`step.F:1248`).

### Summary of the coupling
- The 3-D r.h.s. at n is integrated into `rufrc` (rhs3d), plus horizontal viscosity (uv3dmix), plus surface and bottom stress.
- In step2d, `rufrc − rubar` is AB3-extrapolated and kept constant over the fast steps.
- The fast loop accumulates `Zt_avg1`, `DU_avg1` (primary weights, used for velocity) and `DU_avg2` (secondary weights, used for volume fluxes).
- `set_depth` uses `Zt_avg1`, `set_HUV2` and `step3d_uv2`'s Huon use `DU_avg2`, and `step3d_uv2`'s ū uses `DU_avg1`.

## 2. Configuration mechanism
- **`cppdefs.h`** (2067 lines) is one big `#if defined REGIONAL / #elif defined COASTAL / BASIN / … / SEAGRASS` chain (`cppdefs.h:60, 438, 643 … 2020`). Each case `#define`s physics and numerics keys such as `UV_ADV`, `LMD_MIXING`, `OBC_WEST`, `TS_HADV_UP3`, and the diagnostics.
- **`cppdefs_dev.h`** (1131 lines) is "not for users" (`cppdefs_dev.h:11-17`). It fills in defaults with the "if none chosen, choose X" pattern, e.g. `UV_HADV_UP3` (`:318-333`), `M2FILTER_POWER` (`:107-116`), `TS_HADV`/`UV_VADV` (`:380-418`). It also derives dependent keys such as `NONLIN_EOS → SPLIT_EOS` (`:308-310`) and forces `M2FILTER_NONE` under NBQ or M3FAST (`:144, :239`).
- **`set_global_definitions.h`** holds array-shape macros (`GLOBAL_2D_ARRAY`, `START_2D_ARRAY`, `:47-79`) and logic macros, e.g. `FIRST_TIME_STEP` changes meaning with `EXACT_RESTART` (`:210-218`), and `FIRST_2D_STEP` (`:223-227`).
- **Runtime `croco.in`** is a keyword-block file such as `time_stepping: NTIMES dt NDTFAST NINFO` or `restart: NRST NRPFRST / filename` (`croco.in:3-37`). It is read by `read_inp.F` as an `if/elseif keyword…` chain (106 blocks; e.g. `time_stepping` `:406-430`, `restart` `:682-697`).
- **How incompatible options are detected**, in four layers:
  1. Compile time: a few `#error`s, e.g. `TIDES_MAS` needs `USE_CALENDAR` (`cppdefs_dev.h:592`), and AGRIF cannot be combined with EXACT_RESTART (`:1019`) or with XIOS+OASIS (`:1024`). Mostly though, `cppdefs_dev.h` silently overrides options.
  2. Required keywords in `croco.in`. `checkkwds.F` is a build-time tool that parses `read_inp.F` and generates `setup_kwds.F`, which lists the keywords required by the active cpp keys (`checkkwds.F:10-40`). `read_inp` calls `setup_kwds` (`read_inp.F:316`). Each keyword it parses calls `cancel_kwd` (`check_kwds.F:13-45`), which also catches duplicates. At the end, `check_kwds` (`read_inp.F:3111`; `check_kwds.F:47-66`) reports any keyword that is still missing.
  3. Conflicting switches. `cppcheck.F` is a build-time generator (`Makefile:228-231`) that scans `cppdefs.h` and generates `check_switches1`, which logs all switches and traps more than one project case being defined (`cppcheck.F:13-45`). The hand-written `check_switches2` traps two boundary conditions on the same edge, more than one vertical-mixing scheme, and `ndtfast>1` without SOLVE3D (`check_switches2.F:18-104`).
  4. Errors are counted, not stopped on: every check does `ierr=ierr+1`. They are reported together ("A total of N configuration errors", `read_inp.F:3134-3136`) and stop the run in `main.F:183`.

## 3. State organization, tiling, threading
- State lives in **global common blocks with fixed halos** (`GLOBAL_2D_ARRAY`):
  - `ocean3d.h`: `u,v(…,N,3)`, `t(…,N,3,NT)`, `Hz`, `Hz_bak`, `z_r`, `z_w`, `Huon`, `Hvom`, `We`, `Wi`, `rho` (`ocean3d.h:15-66`);
  - `ocean2d.h`: `zeta`, `ubar`, `vbar` with 4 fast levels (`ocean2d.h:17-18`);
  - `coupling.h`: `rufrc`, `rufrc_bak(…,2)`, `Zt_avg1`, `DU_avg1(…,5)`, `DU_avg2` (`coupling.h:22-39`);
  - `mixing.h`: `Akv(0:N)`, `Akt(0:N,2)`, `visc2_*`, `diff2_*` (`mixing.h:19-96`);
  - `forces.h`: `sustr`, `svstr`, `stflx`, … (`forces.h:19-59`);
  - `grid.h`: `h`, `f`, `pm`, `pn`, … (`grid.h:61-70`).

  Time levels are integer indices that rotate (nstp/nnew/nrhs/indx, kstp/knew). Every routine sees every array.
- **Tiling.** Every physics routine is a thin `xxx(tile)` wrapper that calls `xxx_tile(Istr,Iend,Jstr,Jend, scratch…)`. The bounds come from `compute_tile_bounds.h:33-56` (NSUB_X×NSUB_E chunks; `param.h:245-269`). Scratch is per-thread: `A2d(N2d,NSA,0:NPP-1)`, `A3d` (`private_scratch.h:21-32`), passed as `A2d(1,1,trd)`. MPI halo exchanges (`exchange_*_tile`) sit at the end of each kernel (e.g. `pre_step3d.F:1062-1076`).
- **Threading.** Each OpenMP thread takes a contiguous tile range (`step.F:169-174`). Phases are separated by `C$OMP BARRIER`, and I/O runs under `C$OMP MASTER` (`step.F:605-614`). OpenACC `!$acc kernels` directives are interleaved in the same code.

## 4. Diagnostics (term-by-term budgets)
- **`DIAGNOSTICS_UV`** keeps the 3-D arrays `MXadv`, `MYadv`, `MVadv`, `MCor`, `MPrsgrd`, `MHmix(…,2,2)`, `MHdiff`, `MVmix`, `MVmix2`, `Mrate`, `Mbody`, `MBaro`, `Mfast` (`diagnostics.h:127-142`). Each term is **filled where it is computed, inside the kernel**:
  - `MCor` and `MXadv`/`MYadv` in `rhs3d.F:519-524, 1096-1110`;
  - `MPrsgrd` in `prsgrd.F:352-354, 475-477`;
  - `MHmix` in `uv3dmix_S.F:255, 293`. It is indexed by time level because it is computed in phase 0.
  - `MVmix` is taken before and after the implicit solve (`step3d_uv2.F:163-166, 383-387`);
  - `MBaro` is the coupling correction (`:436-446`);
  - `Mrate` is the Hz·u change over dt (`:786-797`).

  Predictor calls are excluded with `if (nnew.ne.3)` (`rhs3d.F:521, 1100`). `MVmix2` is a **residual closure term**: the rate minus the terms accounted for before coupling, boundary conditions and nudging (`step3d_uv2.F:1403-1407`). `MHdiff` for UP3 is advection minus its centred part, i.e. the implicit hyperdiffusion (`step3d_uv2.F:1409`; `rhs3d.F:1110`). At the end every term is divided by the cell volume, giving m s⁻² (`step3d_uv2.F:1419-1440`).
- **`DIAGNOSTICS_TS`** keeps `TXadv`, `TYadv`, `TVadv`, `THmix`, `TVmix`, `TForc`, `Trate` (`diagnostics.h:16-25`) plus mixed-layer integrals (`_mld`).
  - Filled in `step3d_t.F`: `TXadv` (`:380`), `TVadv` (`:545`), `TForc` (`:621`), `TVmix` (implicit solve), nudging added to `TForc` (`:929`), and `Trate = (Hz·t^{n+1} − Hz_bak·t^n)/(dt·pm·pn)` (`:959`).
  - `DIAGNOSTICS_TS_ADV` adds the divergence correction, so advection is reported in advective rather than flux form (`:964-972`).
  - `t3dmix_S.F` fills `THmix` and **recomputes `Trate`** after the horizontal mixing (`t3dmix_S.F:253-271`).
  - Normalization is at `step3d_t.F:976-982`.
- **EK / VRT / PV** are derived budgets. `set_diags_ek` and `set_diags_vrt` (called in phase 0, `step.F:520-528`, i.e. on the previous step's terms) combine vertically integrated per-term work arrays (`ekwrkHadv`, `ekwrkCor`, …, `set_diags_ek.F:36-60`) into kinetic-energy terms and their curls (`vrtXadv`, `vrtCor`, …, `diags_vrt.h`).
- **Output and averaging.** `wrt_diags`/`wrt_diagsM` and `set_diags*_avg` are called from `output` and from phase 0 (`output.F:296-311`; `step.F:536-560`).

## 5. Restart
- Trigger: `restart: NRST NRPFRST` (`read_inp.F:682-697`). Writing happens when `mod(ilc-1,nrst)==0`. With `EXACT_RESTART` it also happens at `mod(ilc,nrst)==0`, i.e. **two consecutive steps** are saved (`output.F:63-66`, `set_global_definitions.h:210-218`).
- Record index: `record = 1+mod(nrecrst-1, |nrpfrst|)`, so a negative NRPFRST gives a cyclic overwrite (the default −1 keeps only the latest record). With a positive NRPFRST, a new file starts when `record ≥ nrpfrst` (`wrt_rst.F:108-112, 941-943`).
- Contents: ζ and ū (`wrt_rst.F:194-207`), `u/v/t(nstp)` (`:345-372`), KPP `hbls(3−nstp)` (`:389`), GLS `trb`/`Akv` (`:426-450`). Under `EXACT_RESTART`, `rufrc_bak`/`rvfrc_bak(3−nstp)` are also written (`:516-532`). A restart without EXACT_RESTART restarts the AB/LF history with Euler, because FIRST_TIME_STEP is true again.
- `get_initial(nrrec-1,2)` followed by `get_initial(nrrec,1)` reads both levels (`main.F:372-383`). EXACT_RESTART is incompatible with AGRIF (`#error`).

## 6. Lessons for pyOCEAN

**Copy:**
1. **One fixed, documented step order with explicit phases.** CROCO's order is: diagnostic state (ρ, fluxes, ω, K_v) → r.h.s. → predictor → horizontal viscosity → fast mode → geometry update → coupled corrector → tracers → horizontal diffusion → output of the old state. N1's driver should be such a phase list. Where a component may run is a contract: KPP before the r.h.s., GLS after the momentum corrector.
2. **Explicit coupling objects.** CROCO uses `rufrc` (3-D r.h.s. integral minus the 2-D r.h.s., extrapolated and frozen over the fast steps) and three filtered outputs with defined consumers (`Zt_avg1` → geometry, `DU_avg1` → ū, `DU_avg2` → volume fluxes). pyOCEAN should make these a typed `BarotropicForcing`/`BarotropicMeans` object passed between the components. This also fixes the interface for the split-explicit stage C. Note that the momentum r.h.s. integral must include every term that also enters the 3-D update (CROCO adds uv3dmix to `rufrc` by hand).
3. **Budgets filled where the terms are computed, plus a residual.** Each kernel optionally writes its tendency into a named slot, predictor stages are excluded, `rate` is computed from Hz·u before and after, and a residual term (`MVmix2`) closes the budget. pyOCEAN's "term-by-term budgets by default" should copy this, with the registry giving the names and units. Exact closure is then a test.
4. **Configuration checking that collects all errors and reports them together**, with a required-keyword check (`setup_kwds`/`check_kwds`) and mutual-exclusion checks (`check_switches2`). In Python this becomes a typed config with a `validate()` that returns all errors, run as a pre-flight step.
5. **Exact restart that stores all multi-level history** (`rufrc_bak`, two time levels), as pyOCEAN's restart already does.

**Avoid:**
1. **Configuration through cpp.** It makes options compile-time, depends on silent default overrides in `cppdefs_dev.h`, makes the code unreadable (`#if` inside expressions) and makes the option space untestable. Use runtime typed options and component selection instead.
2. **Tiling boilerplate.** This covers the `tile → _tile(Istr,…)` wrappers, per-thread scratch common blocks, and halo exchanges inside every kernel. NumPy, Numba `prange` and JAX don't need it: use whole-array functional kernels and keep halos and decomposition in the driver.
3. **Global mutable state and rotating integer time indices** (nstp/nnew/indx/3). In particular, `u(indx)` changes meaning mid-step (velocity, then the Hz·u accumulator). Prefer explicit state objects (`state_n`, `state_half`, tendencies) with pure kernels. That is also what JAX needs.
4. **A step re-entered through a counter (`iif`).** It is clever for AGRIF but hides the control flow. A plain loop over the fast steps inside a barotropic component is clearer, and nesting hooks can go between phases.
5. Diagnostics hidden behind `#if` and also mixed into kernel loops. In pyOCEAN, a kernel should return its tendency and the driver should decide whether to store it, so there are no diagnostic branches inside Numba loops.
