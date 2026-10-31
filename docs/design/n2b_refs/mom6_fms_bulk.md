# MOM6 + FMS: cómo se calculan los flujos aire–mar desde una reanálisis

Lectura de fuente para N2b (flujos de superficie completos en 3-D). Rutas relativas a `~/pyOCEAN_refs/`.
Marca **[verificar]** = no comprobado en código local.

## 0. Hallazgo principal: el bulk NO está en el código local

MOM6 no calcula bulk. En la configuración acoplada GFDL (OM4, regionales tipo NWA12) el cálculo lo hace
el **modelo de hielo SIS2** (componente "ICE" del acoplador) con `surface_flux_mod` de **FMScoupler**
(`ncar_ocean_fluxes`). FMScoupler e `ice_param` no venían con MOM6-examples; desde N2b tarea 1 (2026-10-31) están clonados de
GitHub fuera del repo: `~/pyOCEAN_refs/FMScoupler` (NOAA-GFDL, **tag 2025.02, git 5b700a4**, la versión de FMS 2025.02; el FMS local es
2025.02.01) e `~/pyOCEAN_refs/ice_param` (git 1d19a57, 2024-06-12; sin tags 2025). Rutas de esta sección y la 2: `FMScoupler/shared/surface_flux.F90`,
`ice_param/{ocean_albedo,ocean_rough}.F90`. Antes de clonarlos:

- `MOM6-examples/src/{coupler,FMS,FMS1,FMS2,ice_param,SIS2,atmos_null,...}` son submódulos **vacíos** (0 ficheros).
- `FMS/exchange/` solo contiene `xgrid.F90` y `stock_constants.F90`; no hay `surface_flux*.F90`,
  `ocean_albedo*.F90` ni `ocean_rough*.F90` (grep sobre FMS, MOM6, MOM6-examples).
- `FMS/monin_obukhov/` existe. `surface_flux_1d` llama siempre a `mo_drag` (`surface_flux.F90:386-388`), pero sobre agua
  `ncar_ocean_fluxes` sobrescribe `cd_m, cd_t, cd_q, u_star, b_star` (`:401-405`): ni `monin_obukhov_nml neutral` ni la rugosidad
  Beljaars de `ocean_rough_nml` cambian los flujos oceánicos (la rugosidad solo entra en `iter_monin_obukhov_ocean`, apagado, `:390-399`).
- `MITgcm/pkg/atm_phys/surface_flux_mod.F90` es un modelo atmosférico, no es lo que usa MOM6.

Lo que sí hay en disco y está verificado: la interfaz MOM6<->coupler (`config_src/drivers/FMS_cap/`), el
estado superficial (`src/core/MOM.F90`), la aplicación de flujos (`MOM_forcing_type.F90`,
`MOM_diabatic_aux.F90`), la óptica (`MOM_opacity.F90`), `data_override` y `time_interp` de FMS, y los
ficheros de configuración OM4_025 (`input.nml`, `data_table`, `MOM_input`, `*_parameter_doc.all`).
El algoritmo bulk (sec. 2) está ahora leído en `FMScoupler/shared/surface_flux.F90` (no `full/`), portado a
`forcing/bulk.py::bulk_ncar_fms` e igual bit a bit al compilado sobre 250 puntos (`validation/bulk_points/`).

Existe además un camino solo-oceánico: `config_src/drivers/solo_driver/MOM_surface_forcing.F90`. No hace
bulk tampoco; lee flujos ya formados (stress, buoy, restoring) por `data_override`/archivos.
`docs/design/era5_bulk_comparison.md` ya resume esto en una línea (fila "MOM6").

## 1. Estado superficial que ven los flujos

`extract_surface_state` (`MOM6/src/core/MOM.F90:4136-4536`):

- **No hay mixed-layer bulk en OM4** (`BULKMIXEDLAYER` no activo; ALE). Con `CS%Hmix >= 0` (rama `MOM.F90:4223`, else en 4242)
  SST y SSS son el **promedio ponderado por espesor sobre los primeros `HMIX_SFC_PROP` metros**
  (`MOM.F90:4242-4301`): cada capa aporta `dh = min(h_k, depth_ml - depth)`; luego `SST = Σ dh·T / depth`.
- Default `HMIX_SFC_PROP = 1.0 m` (`MOM.F90:2695-2699`); OM4_025 usa el default
  (`MOM6-examples/ice_ocean_SIS2/OM4_025/MOM_parameter_doc.all:104`). Es una profundidad fija, **independiente de la
  capa**: con capa superior fina promedia varias capas; con capa superior gruesa (>1 m) toma solo la
  fracción de 1 m de esa capa, es decir su valor (capa homogénea). Nunca interpola dentro de la capa.
- Si `depth(i) < H_subroundoff` (columna casi vacía) usa `T(k=1)` (`MOM.F90:4271-4300`, rama `answer_date >= 20190101`);
  OM4_025 usa `DEFAULT_ANSWER_DATE = 20181231` (doc.all:179), rama antigua que divide por
  `max(depth, H_subroundoff)`.
- **Velocidad superficial**: `HMIX_UV_SFC_PROP = 0` por defecto (`MOM.F90:2700-2704`) => sin promedio: `u,v` de la
  **capa 1** (`MOM.F90:4360-4367`). Si fuera >0 promedia con espesores de cara `hu = ½(h_i + h_{i+1})`
  (`MOM.F90:4310-4358`). Observación: en el bucle de `u` (línea 4345) el test usa `depth(i)` en minúscula
  dentro de un bucle sobre `I`; parece un descuido inocuo para `HMIX_UV>0` [verificar], no aplica con el default.
- Conversión para el acoplador (`FMS_cap/ocean_model_MOM.F90:870-897`): `t_surf = SST + 273.15` (si el modelo usa
  conservative T, antes `gsw_pt_from_ct` => **temperatura potencial**), `s_surf` practical salinity si
  la S es absoluta. `u_surf,v_surf` de `sfc_state%u,v` (líneas 933-947). El acoplador usa estos valores
  (promediados en el paso de acoplamiento, `dt_cpld = 3600 s`, `OM4_025/input.nml`) en el bulk.
- **Frazil**: `FRAZIL = True` (MOM_input:60). `make_frazil` (`MOM_diabatic_aux.F90:120-240`) enfría a punto
  de congelación `T_f` (dependiente de presión, `PRESSURE_DEPENDENT_FRAZIL = True`) y acumula el calor en
  `sfc_state%frazil` [J m-2], que se devuelve al modelo de hielo (`ocean_model_MOM.F90:913-915`). `HFREEZE = -1` => sin
  `melt_potential` (`MOM.F90:2706-2711`). Irrelevante para una región sin hielo, pero fija T >= T_f.
- Otros: `BOUND_SALINITY = True`, `MIN_SALINITY = 0.01` (MOM_input:66,69).

## 2. Algoritmo bulk (FMS `surface_flux_mod`, NO verificable localmente)

Configuración verificada en `OM4_025/input.nml:114-115` (idéntica en OM_1deg, OM4_05, Baltic*, OM4_025.JRA):

```
&surface_flux_nml  ncar_ocean_flux = .true.   raoult_sat_vap = .true. /
&monin_obukhov_nml neutral = .true. /      &ocean_rough_nml rough_scheme = 'beljaars' /
&ocean_albedo_nml  ocean_albedo_option = 2 /   &ice_albedo_nml t_range = 10. /
```

Alturas y entradas, verificado en `OM4_025.JRA/data_table` y `OM4_025/data_table`:
`z_bot = 10.0 m` (constante), `t_bot = tas` (JRA)/`T_10_MOD` (CORE), `sphum_bot = huss`/`Q_10_MOD`,
`u_bot,v_bot = uas,vas`/`U_10_MOD`, `p_bot = p_surf = psl`, `gust = 1.0e-4` constante.
=> **Todas las variables atmosféricas se suponen a 10 m**: no hay ajuste 2 m -> 10 m en MOM6/FMS; la
preparación (p. ej. ERA5 2 m -> 10 m) es responsabilidad del conjunto de forzamiento (CORE ya está normalizado a 10 m).
Interpolación espacial `bilinear` (`bicubic` en u,v).

Del código (`FMScoupler/shared/surface_flux.F90`, tag 2025.02; verificado y compilado, `validation/bulk_points/`):
- **Rutinas**: `surface_flux_1d` (`:216-500`) prepara el estado y forma los flujos; con `ncar_ocean_flux = .true.` llama a
  `ncar_ocean_fluxes` (`:774-897`, todo a la altura `z_atm`); con `ncar_ocean_flux_multilevel = .true.` a
  `ncar_ocean_fluxes_multilevel` (`:905-1044`, R. Dussin 2020, alturas `bulk_zu/zt/zq` de la namelist, `:185-187`, `:407-415`), que
  además lleva T y q de z_t, z_q a z_u (LY04 9b-c) y los devuelve para los flujos. OM4 usa el de un nivel. Con z_t = z_q = z_u ambos
  dan lo mismo bit a bit. `ncar_ocean_flux_orig` conserva el error histórico (C_h, C_e elevados al cuadrado, `:835-836`); OM4 usa el
  corregido (`:889-890`, "Jul2007 bugfix").
- **Coeficientes**: C_dN = (2.7/U₁₀ + 0.142 + **0.0764** U₁₀)/10³ (`:849`; LY04 imprime U/13.09, CESM usa 0.0764); C_eN = 34.6 √C_dN/10³;
  C_hN = (18.0 estable | 32.7 inestable) √C_dN/10³ (`:851-853`). Primera estimación con U₁₀ = U y estabilidad por el signo de T_a − SST.
- **Iteraciones**: `n_itts = 2` (`:784`, `:936`), sin tolerancia.
- **ψ y ζ**: ζ = κ b* z/u*², b* = g (T*/T_v + q*/(q + 1/0.608)) con T_v = T(1 + 0.608 q) (`:845`, `:863-864`); recorte |ζ| ≤ 10 (`:865`,
  "undocumented NCAR"); X² = max(√|1 − 16ζ|, 1) (`:866-867`); ψ_m = ψ_h = −5ζ estable, Paulson inestable (`:870-876`). Estable si ζ > 0
  (un nivel, `:870`) o ζ ≥ 0 (multinivel, `:994`); en ζ = 0 da lo mismo. La C_hN de la iteración usa `sign(0.5, ζ)` ⇒ ζ = 0 cuenta como estable (`:882`).
- **Viento mínimo**: U = max(W, 0.5 m/s) **solo en los coeficientes** (`:846`); los flujos usan W sin piso (`:423-425`). U₁₀ no tiene piso (`:878`).
- **Ráfagas**: W = √(Δu² + Δv² + gust²) (`:378`), `gust` del data_table (OM4: constante 1e-4 m/s, campo sin fichero con factor 1e-4);
  `gust_min = 0` por defecto (`:171`, `:367-374`), `alt_gustiness = .false.` (`:164`). Efecto ≤ 5e-7 relativo a 0.1 m/s.
- **Viento relativo**: Δu = u_surf − u_atm (`:349-350`), con u_surf de MOM6 (capa 1, sec. 1).
- **Humedad de saturación**: `escomp` = `lookup_es` (FMS `sat_vapor_pres.F90:273-278`), la tabla **TABLE** de −160 a 100 °C cada 0.1 K
  (`:759-761`, `:823`) con interpolación cuadrática (`sat_vapor_pres_k.inc:1518-1554`); TABLE es Goff–Gratch sobre agua por encima de
  273.16 K, sobre hielo por debajo de 253.16 K y mezcla lineal entre ambas (`sat_vapor_pres_k.inc:216-289`): a −1.8 °C da 0.16 % menos
  que sobre agua. `construct_table_wrt_liq = .true.` (OM4) solo construye TABLE2 para `lookup_es2` y **no** cambia este cálculo.
  q_sat = ε e/(p_surf − (1 − ε) e), ε = R_d/R_v (`:320`), × 0.98 con `raoult_sat_vap` (`:331`).
- **Temperatura del aire**: θ_a = T_a (p_surf/p_atm)^κ (`:341`, `:344`); en OM4 p_bot = p_surf = psl (data_table), así que θ_a = T_a
  (sin corrección Γz).
- **Densidad del aire**: ρ = p_atm/(R_d T_v), T_v = T_a(1 + d608 q), d608 = (1 − ε)/ε = 0.6078 (`:343`, `:428`); dentro de la
  iteración el T_v usa 0.608 (`:845`).
- **Constantes** (`FMS/constants/gfdl_constants.fh`): `GRAV = 9.80`, `VONKARM = 0.40`, `RDGAS = 287.04`, `RVGAS = 461.5`,
  `KAPPA = 2/7`, `CP_AIR = RDGAS/KAPPA = 1004.64`, `HLV = 2.5e6`, `HLF = 3.34e5`, `STEFAN = 5.6734e-8`, `TFREEZE = 273.16`.
- Gustiness de MOM6: `GUST_CONST` (sec. abajo) es otra cosa; solo afecta a u* para mezcla.
- **Salidas** (`:423-461`): `flux_u = ρ C_d W (u_surf − u_atm)` (stress **sobre el aire**; sobre el océano es −flux_u),
  `flux_t = c_p ρ C_h W (T_s − θ_a)` y `flux_q = ρ C_e W (q_s − q_a)` positivos hacia arriba, `flux_r = σ T_s⁴` (ε = 1, `:454-455`),
  `u_star` = √C_d U con la C_d del paso anterior de la iteración (`:859-860`). Las derivadas `dhdt_surf`, `dedt_surf`, `drdt_surf`
  (`:433-456`) son para el acoplamiento implícito con la atmósfera; los valores `"ICE" dhdt=80, dedt=2e-6, drdt=10` del data_table
  (OM4_025/data_table:18-20) no los lee `surface_flux.F90` [verificar su uso en el acoplador o SIS2].

Conversión en MOM6 (verificada, `FMS_cap/MOM_surface_forcing_gfdl.F90`):
- `fluxes%sens = - t_flux` (líneas 560-561); `fluxes%evap = - q_flux` (496-497);
  `fluxes%latent = - q_flux · latent_heat_vapor` (576-578) con `LATENT_HEAT_VAPORIZATION` default `hlv = 2.5e6`
  (1436-1437; `MOM_constants.F90` de `config_src/infra/FMS2` toma `HLV` de FMS). Más `- fprec·hlf` (567-569).
  Convención MOM: positivo = hacia el océano; `sens` y `latent` típicamente negativos (`MOM_forcing_type.F90:112-116`).
- Stress: `taux,tauy` de `IOB%u_flux,v_flux` (`extract_IOB_stresses`, líneas 971-1218), `WIND_STAGGER = C`, multiplicador 1.
  `ustar` de la magnitud del stress más gustiness `GUST_CONST` (OM4_025: `GUST_CONST = 0.0`, doc.all:2736; default del cap FMS 0.0,
  `MOM_surface_forcing_gfdl.F90:1696-1699`; en el solo_driver el default es 0.02 Pa) (líneas 1123-1135). Con `READ_GUST_2D` puede ser campo 2-D.

## 3. Radiación

- **Entrada**: `lw_flux_dn` (rlds), y SW en cuatro bandas desde un único campo `rsds` con factores en data_table:
  `sw_flux_vis_dir_dn`, `_vis_dif_dn` ×0.285 cada una, `nir_dir`, `nir_dif` ×0.215 (`OM4_025.JRA/data_table`) => 57 % visible / 43 % NIR,
  directo y difuso iguales. SIS2 añade ciclo diurno sintético (`ADD_DIURNAL_SW = True`, `SIS_parameter_doc.all:68`) porque estos campos son
  diarios en OM4; `DT_RADIATION = 3600`.
- **Albedo oceánico**: se aplica **en el modelo de hielo/acoplador**, no en MOM6 (MOM recibe SW ya neto: `fluxes%sw = vis_dir + vis_dif + nir_dir + nir_dif`,
  `MOM_surface_forcing_gfdl.F90:583-608`). `ocean_albedo_option = 2` (`input.nml`) y `DO_SUN_ANGLE_FOR_ALB = False` (SIS doc.all:70).
  Opción 2 (`ice_param/ocean_albedo.F90:56-58`, "GFDL Climate Dynamics Group in 80s and 90s, source not currently documented"): albedo
  función solo de la latitud, igual en las cuatro bandas (`:362-367`), interpolado linealmente cada 10° en la tabla `albedo_mcm`
  (`:153-154`, `:501-510`): 0.206, 0.161, 0.110, 0.097, 0.089, 0.076, 0.068, 0.063, 0.060 (10°S–10°N), de 90°S al ecuador, y simétrico; no es Large–Yeager.
- **LW**: `fluxes%LW = lw_flux` del hielo (líneas 554-557), neta hacia el océano ("typically negative", MOM_forcing_type.F90:106). El cálculo
  `ε(LW↓ − σT_s⁴)` se hace en SIS2 con `T_s = t_surf` (SST potencial, K). `surface_flux_1d` calcula `flux_r = σ T_s⁴` con ε = 1
  (`surface_flux.F90:454-455`); qué emisividad usa SIS2 al formar la LW neta del océano queda [verificar] (SIS2 no está en disco).
- **Penetración de SW en MOM6** (verificado): `OPACITY_SCHEME = "MANIZZA_05"`, `VAR_PEN_SW = True`, `CHL_FROM_FILE` (SeaWiFS climatología),
  `PEN_SW_NBANDS = 3` (MOM_parameter_doc.all:2306-2309, 2475-2509). `MOM_opacity.F90:352-380`: banda 1 azul = `BLUE_FRAC_SW (0.5)·SW_vis`, banda 2 roja = `(1−0.5)·SW_vis`,
  banda 3 NIR = `SW_nir`. Opacidades `a = c1 + c2·Chl^p`: azul (0.0232, 0.074, p=0.674), rojo (0.225, 0.037, p=0.629), NIR (2.86, 0.0) m-1
  (`OPACITY_VALUES_MANIZZA`, `CHOROPHYLL_POWER_MANIZZA`, :1189-1201). El resto no penetrante/NIR se absorbe en las capas superficiales
  (`PEN_SW_FLUX_ABSORB 2.5e-11`, `PEN_SW_ABSORB_MINTHICK 1 mm`, `absorbRemainingSW`, `MOM_opacity.F90:602-869`).
  Si solo hay SW total: 42 % visible (`:361`). Lo no-penetrante (calor turbulento + LW + SW absorbido arriba) entra en la capa superior con
  `minimum_forcing_depth` = 1 mm (`MINIMUM_FORCING_DEPTH = 0.001`, doc.all:1934), repartiendo hacia abajo en capas finas.
  Net: `net_heat = dt/(Cp ρ) (sw + lw + latent + sens [+ heat_added])` (`MOM_forcing_type.F90:709-727`), con el SW penetrante
  descontado en `Pen_SW_bnd` (`:606-614`). `C_P = 3992` (MOM_input:72), `RHO_0 = 1035`.

## 4. Agua dulce: P, E, runoff y restoring

- Todo en kg m-2 s-1 hacia el océano (`MOM_forcing_type.F90:125-132`): `lprec` (prra), `fprec` (prsn), `evap = −q_flux` (<0 = evaporación), `lrunoff`
  (`friver`), `frunoff` (`licalvf`), `vprec` (virtual del restoring). Los datos de entrada ya son **flujos medios** (kg m-2 s-1), factor 1.0 en data_table.
- **Es flujo de agua real** (volumen/masa): `netMassInOut = dt·(lprec + fprec + evap + lrunoff + vprec + ...)` (`MOM_forcing_type.F90:631-641`) cambia el espesor de la
  capa superior (ALE); `netMassOut` (evap <0, lprec <0) se retira de capas con `EVAP_CFL_LIMIT = 0.8`, y se suma la masa entrante a la **T de la capa superior** y
  **S = 0** (`MOM_diabatic_aux.F90:1020-1043`, `Temp_in = T2d(i,k)`, `Salin_in = 0`); la masa saliente se saca a T local (`:1098-1125`). El calor de masa
  (lprec, runoff, etc. a SST) no genera flujo adicional salvo `useRiverHeatContent` (opcional, `:~715-735`).
- **Restoring de salinidad (OM4)**: `RESTORE_SALINITY = True`, `FLUXCONST = 0.1667 m/día` (= `FLUXCONST_SALT`), `SALT_RESTORE_FILE` PHC2.1,
  `SRESTORE_AS_SFLUX = True` (flujo de sal, no agua virtual), `MAX_DELTA_SRESTORE = 5`, `ADJUST_NET_SRESTORE_TO_ZERO = True`,
  `ADJUST_NET_FRESH_WATER_TO_ZERO = True` (doc.all:2651-2711). Flujo: `salt_flux = 1e-3·ρ_restore·Flux_const·ΔSSS` con
  `ΔSSS` recortado a ±`MAX_DELTA_SRESTORE`, `RESTORE_FLUX_RHO = 1035` (`MOM_surface_forcing_gfdl.F90:380-420`). Como agua virtual (`SRESTORE_AS_SFLUX=False`):
  `vprec = ρ·F·ΔSSS / (½(SSS+S_data))` (`:436-440`). `RESTORE_TEMPERATURE = False` por defecto; si se activa: `heat_added = ρ Cp ΔSST F_T` con
  ΔSST recortado a `MAX_DELTA_TRESTORE` (`:425-431`). Defaults: ambos False (flags en `solo_driver` `RESTOREBUOY`).
  El restoring de OM4 se describe como "CORE mode, no usar en el CM4 acoplado" (MOM_saltrestore:1-2).
- `ADJUST_NET_FRESH_WATER_TO_ZERO` ajusta `vprec` para que el agua neta global sea 0 (`:659-690`).
- Frazil/sal de hielo: `salt_flux` del hielo se suma al flujo de sal (`:634-640`).

## 5. Interpolación temporal (FMS `data_override`)

- Cada entrada de data_table: `fieldname_code, file, fieldname_file, interp_method, factor` (`data_override.inc:69, 619-646`). El `factor` solo
  multiplica el campo (`data_out*factor`, :1007, 1645, ...): **no hay conversión de acumulados**. Los campos deben llegar ya como tasas
  (W m-2, kg m-2 s-1, m s-1...). ERA5 acumulado (J m-2 por hora, m de agua) hay que dividirlo por 3600 s (y x1000 kg m-3 para precip.) **antes**, o usar
  un `factor` que sirva solo para una constante (no depende de dt variable). Inferido: ningún código de desacumulación en data_override/time_interp.
- Tiempo: `time_interp_external` (`FMS/time_interp/include/time_interp_external2.inc:140-230`): con un solo registro lo devuelve constante;
  si no, **interpolación lineal entre los dos registros vecinos** `data = d1·(1−w2) + d2·w2` con `w2 = (T − t1)/(t2 − t1)` (`time_interp.inc:345`, `:223-228`
  del .inc de external). Los instantes son los del eje `time` del fichero, tomados como instantes puntuales (sin promedio centrado en el intervalo:
  un acumulado/promedio horario etiquetado al final del intervalo se interpretaría como instantáneo en esa hora). Soporta climatología modulo (`modulo_time`).
  Si un registro está enmascarado, el resultado es `missing`.
- Espacial: `bilinear`/`bicubic` (horiz_interp) o `none` (ongrid, mismo grid) (`data_override.inc:436-445, 1283-1429, 1475-1534`).
- Paso de acoplamiento de OM4: `dt_cpld = 3600`, bulk con SST de ese instante (no hay subpasos).

## 6. Configuraciones del ejemplo (resumen de ajustes típicos)

| Ajuste | Valor OM4_025 / JRA | Fuente |
|---|---|---|
| bulk | NCAR (`ncar_ocean_flux`), Raoult 0.98 | `input.nml:114-115` |
| rugosidad | Beljaars (`ocean_rough_nml`), MO `neutral=.true.`: sin efecto sobre agua | `input.nml`; `surface_flux.F90:384-405` |
| alturas | z_bot = 10 m, todo a 10 m | `data_table` |
| albedo | `ocean_albedo_option = 2` | `input.nml:103-105` |
| SW bandas | 0.285 vis dir, 0.285 vis dif, 0.215 nir dir, 0.215 nir dif | `OM4_025.JRA/data_table` |
| ciclo diurno | `ADD_DIURNAL_SW` en SIS2 | SIS doc.all:68 |
| precip. | `prra`, `prsn` (flujos), factor 1 (CORE: ×0.9933) | data_table |
| runoff | `friver` sin interpolación (`none`), calving aparte | data_table |
| SSS restoring | sal flux, 0.1667 m/día, PHC2.1, max 5 | `MOM_saltrestore` |
| SW penetrante | MANIZZA_05, VAR_PEN_SW, 3 bandas, Chl SeaWiFS | doc.all:2306, 2475-2509 |
| SST/SSS al bulk | promedio de 1 m (`HMIX_SFC_PROP`) | `MOM.F90:2695` |
| `HMIX_UV_SFC_PROP` | 0: capa 1 | `MOM.F90:2700` |
| `C_P`, `RHO_0`, `HLV` | 3992, 1035, 2.5e6 | `MOM_input:72`, doc.all:349 |

No hay ejemplo regional (NWA12) en disco: solo OM4_*, Baltic_OM4_* (misma configuración de flujos) y `ocean_only/` (flujos de archivo).

## 7. Comparación con `bulk_ncar` de pyOCEAN (`forcing/bulk.py::bulk_ncar`)

Mismo linaje (Large & Yeager 2004, ecs. 4-10, 2 iteraciones, |ζ| ≤ 10). Verificado en N2b tarea 1 con la rutina compilada:
`forcing/bulk.py::bulk_ncar_fms` es el port de FMS (= compilado bit a bit, `tests/test_bulk_fms.py`); las diferencias de
`bulk_ncar` medidas sobre los 250 puntos de `validation/bulk_points/points.py` (todo a 10 m; max|Δ|/max|FMS|, cada una sola) están en
`validation/bulk_points/README.md` y `points.LY04_VS_FMS_MAX_REL`. Todas son (c), elecciones entre formas publicadas:

| Aspecto | FMS/MOM6 (OM4) | pyOCEAN `bulk_ncar` | Efecto |
|---|---|---|---|
| Fórmulas C_dN, C_eN, C_hN | LY04 6a-c con pendiente 0.0764 (`:849`) | 1/13.09 | τ 7e-5 |
| Iteraciones | 2 (`:784`) | `n_iter=2` | igual |
| Alturas | `ncar_ocean_fluxes` todo a `z_atm` (OM4: 10 m); multinivel opcional con 9b-c | `z_u=10`, `z_t=2` por defecto, corrección 9b-c | igual con alturas iguales |
| Temp. del aire | θ_a = T_a (p_atm = p_surf) | θ = T + 0.0098 z_t | Q_S 1.0e-2 |
| Viento en los flujos | W sin piso; 0.5 m/s solo en coeficientes | max(W, 0.5) también en los flujos | Q_S 3.8e-2, Q_L 4.8e-2 (W = 0.1 m/s) |
| Piso de U₁₀N | no | 0.5 m/s | Q_L 3.6e-4 |
| Ráfagas | √(W² + gust²), gust 1e-4 m/s en OM4 | no | ≤ 5e-7 |
| q_sat | tabla FMS Goff–Gratch (mezcla con hielo < 0.01 °C) × 0.98 | LY04 eq. 5 × 0.98 | q_s +0.04 … +2.2 %; Q_L 5.8e-2 |
| cp_air | 1004.64 | 1000.5 | Q_S 4.1e-3 |
| g | 9.80 | 9.81 | ≤ 4e-5 |
| ρ_aire | p/(R_d T(1 + 0.6078 q)) | p/(R_d θ(1 + 0.608 q)) | ≤ 5e-6 |
| L_v | 2.5e6 | 2.5e6 | igual |
| SST en el bulk | temperatura potencial (promedio 1 m) | SST dado por el llamador | alinear con sec. 1 |
| Signos | sens=−t_flux, evap=−q_flux (MOM) | positivo hacia el océano | mismo convenio final |
| Albedo | opción 2: tabla por latitud (sec. 3), fuera de MOM6 | `radiative_fluxes`: 0.066 o `albedo_large_yeager` | revisar en la tarea de radiación |
| LW | σT⁴ con ε = 1 en `surface_flux`; neta de SIS2 [verificar ε] | ε(LW↓−σT⁴), ε=0.97 | |

Conclusión: con las nueve diferencias activadas una a una, un modelo escalonado reproduce `bulk_ncar` desde FMS a 6e-16, así que la lista
es completa. `bulk_kara` (HYCOM flxflg=6) es otro algoritmo y no aplica a MOM6.

## 8. Implicaciones para N2b

1. Implementar el camino MOM6 como tres bloques separados (como MOM6): (a) bulk NCAR a 10 m -> (stress, sens, evap); (b) conversión (`sens`, `latent = evap·L_v`, `lprec/fprec/lrunoff`);
   (c) aplicación en el océano (calor neto, SW penetrante por bandas, agua real que entra a T_top con S=0).
2. SST/SSS al bulk: promedio de 1 m de espesor (no la capa superior ni el valor del nivel 1) y velocidad = capa 1.
3. Forzamiento ERA5: desacumular antes, no usar `factor` para eso; interpolación lineal en el tiempo entre instantes etiquetados.
4. Penetración SW: MANIZZA_05, 3 bandas, 57 % vis / 43 % NIR, `BLUE_FRAC_SW = 0.5`; el albedo se debe aplicar antes.
5. Restoring de S: opcional, como sal flux, con tope ±5 y balance neto cero; OFF por defecto para regionales con OBC.

## [verificar] resumen
Resuelto en N2b tarea 1 (2026-10-31, FMScoupler 2025.02 e ice_param leídos y `surface_flux.F90` compilado): todo `ncar_ocean_fluxes`
(iteraciones, ψ, recortes, viento mínimo, ráfagas, T potencial, ρ_aire, q_sat), `ocean_albedo_option = 2`, `ocean_rough_nml`/`monin_obukhov_nml`
(sin efecto sobre agua) y cómo entra `gust`. Queda:
- Emisividad de la LW neta en SIS2; promedio temporal de SST/u_surf en el paso de acoplamiento (SIS2 y `coupler_main` no leídos).
- Uso de `dhdt, dedt, drdt` (flujo implícito) del data_table en el acoplador.
- Posible uso inconsistente de `depth(i)` vs `depth(I)` en `MOM.F90:4345` con `HMIX_UV_SFC_PROP > 0`.
- Si `useRiverHeatContent` se activa en OM4 y el tratamiento exacto del calor del runoff.
