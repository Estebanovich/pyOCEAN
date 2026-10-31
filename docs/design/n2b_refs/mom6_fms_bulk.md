# MOM6 + FMS: cómo se calculan los flujos aire–mar desde una reanálisis

Lectura de fuente para N2b (flujos de superficie completos en 3-D). Rutas relativas a `~/pyOCEAN_refs/`.
Marca **[verificar]** = no comprobado en código local.

## 0. Hallazgo principal: el bulk NO está en el código local

MOM6 no calcula bulk. En la configuración acoplada GFDL (OM4, regionales tipo NWA12) el cálculo lo hace
el **modelo de hielo SIS2** (componente "ICE" del acoplador) con `surface_flux_mod` de **FMScoupler**
(`ncar_ocean_fluxes`). Ni FMScoupler ni SIS2 ni `ice_param` están en disco:

- `MOM6-examples/src/{coupler,FMS,FMS1,FMS2,ice_param,SIS2,atmos_null,...}` son submódulos **vacíos** (0 ficheros).
- `FMS/exchange/` solo contiene `xgrid.F90` y `stock_constants.F90`; no hay `surface_flux*.F90`,
  `ocean_albedo*.F90` ni `ocean_rough*.F90` (grep sobre FMS, MOM6, MOM6-examples).
- `FMS/monin_obukhov/` existe (otra rutina, no usada por `ncar_ocean_flux=.true.` [verificar]).
- `MITgcm/pkg/atm_phys/surface_flux_mod.F90` es un modelo atmosférico, no es lo que usa MOM6.

Lo que sí hay en disco y está verificado: la interfaz MOM6<->coupler (`config_src/drivers/FMS_cap/`), el
estado superficial (`src/core/MOM.F90`), la aplicación de flujos (`MOM_forcing_type.F90`,
`MOM_diabatic_aux.F90`), la óptica (`MOM_opacity.F90`), `data_override` y `time_interp` de FMS, y los
ficheros de configuración OM4_025 (`input.nml`, `data_table`, `MOM_input`, `*_parameter_doc.all`).
Todo lo del algoritmo bulk (sec. 2) es de memoria o deducido de namelists: [verificar] contra
`NOAA-GFDL/FMScoupler`, `full/surface_flux.F90` (`ncar_ocean_fluxes`).

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

Del código de `ncar_ocean_fluxes` (de memoria, es la rutina de CESM/`shr_flux_atmOcn` de Large & Yeager 2004 adaptada por GFDL):
[verificar] todos los puntos siguientes.
- Coeficiente neutro C_dN10 = (2.7/U10 + 0.142 + U10/13.09)·1e-3; C_eN = 34.6e-3 √C_dN;
  C_hN = (18.0 estable | 32.7 inestable)·1e-3 √C_dN (LY04 eqs 6a-c).
- Corrección de estabilidad con ψ_m, ψ_h de LY04 (Paulson inestable, −5ζ estable), ζ de u*, b*; **2 iteraciones** (`n_itts = 2`).
- Viento relativo a la corriente: `u_atm - u_surf` (el `u_surf` del sec. 1) [verificar].
- Humedad de saturación: `sat_vapor_pres` (tabla respecto a líquido, `sat_vapor_pres_nml construct_table_wrt_liq = .true.`,
  verificado en input.nml) y factor **0.98** (Raoult) por `raoult_sat_vap = .true.`.
- Constantes FMS verificadas en `FMS/constants/gfdl_constants.fh`: `RDGAS = 287.04`, `RVGAS = 461.5`,
  `KAPPA = 2/7`, `CP_AIR = RDGAS/KAPPA = 1004.64`, `HLV = 2.5e6`, `HLF = 3.34e5`, `STEFAN = 5.6734e-8`.
  Densidad del aire: gas ideal con T_v [verificar].
- Gustiness: la tabla `gust = 1e-4` entra en el viento efectivo [verificar si se suma en cuadratura o es `gust_min`]. Es distinto de
  `GUST_CONST` de MOM6 (sec. abajo), que solo afecta a u* para mezcla.
- Salidas del bulk al acoplador: `u_flux, v_flux` (stress sobre el océano), `t_flux` (sensible, positivo hacia arriba),
  `q_flux` (evaporación, kg m-2 s-1, positivo hacia arriba), más derivadas implícitas
  (`dhdt=80, dedt=2e-6, drdt=10` en data_table) [verificar su uso].

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
  `MOM_surface_forcing_gfdl.F90:583-608`). `ocean_albedo_option = 2` (`input.nml`) y `DO_SUN_ANGLE_FOR_ALB = False` (SIS doc.all:70). Esquema y
  valores de la opción 2 [verificar] (código `ocean_albedo` ausente; no asumir que sea Large-Yeager).
- **LW**: `fluxes%LW = lw_flux` del hielo (líneas 554-557), neta hacia el océano ("typically negative", MOM_forcing_type.F90:106). El cálculo
  `ε(LW↓ − σT_s⁴)` se hace en SIS2 con `T_s = t_surf` (SST potencial, K) [verificar emisividad y exactitud, p. ej. 0.97 vs 1].
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
| rugosidad | Beljaars (`ocean_rough_nml`), MO `neutral=.true.` | `input.nml` [verificar uso] |
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

## 7. Comparación con `bulk_ncar` de pyOCEAN (`forcing/bulk.py:236-432`)

Mismo linaje (Large & Yeager 2004, ecs. 4-10, 2 iteraciones, clip |ζ|<=10 en pyOCEAN). La comparación de FMS es **de memoria** salvo donde se indica:

| Aspecto | FMS/MOM6 (OM4) | pyOCEAN `bulk_ncar` | Diferencia |
|---|---|---|---|
| Fórmulas C_dN, C_eN, C_hN | LY04 6a-c [verificar] | idénticas (`cd_neutral_ly04`, 34.6e-3, 18.0/32.7e-3) | ninguna esperada |
| Iteraciones | 2 [verificar] | `n_iter=2` | igual |
| Alturas | todo a 10 m (data_table verificado) | `z_u=10`, `z_t=2` por defecto, con corrección de θ,q de z_t a z_u (9b,9c) | usar `z_t=10` para igualar; con 2 m ERA5 pyOCEAN corrige perfil, FMS no |
| Temp. del aire | pasada a potencial vía presión [verificar] | θ = T + Γ z_t (Γ=0.0098) | diferencia pequeña a 10 m |
| Viento relativo | u_atm − u_surf (u_surf capa 1) | `u_ocean`, `v_ocean`; sin gust | usar capa 1 (HMIX_UV=0); FMS añade `gust` [verificar] y velocidad mínima [verificar] |
| Mínimo de viento | [verificar] | `u_min=0.5` | |
| q_sat | tabla `sat_vapor_pres` sobre líquido × 0.98 | LY04 eq. 5 `640380/ρ exp(−5107.4/T)` × 0.98 | forma funcional distinta (hasta ~1 % en q_s) |
| cp_air | 1004.64 (RDGAS/KAPPA, verificado) | 1000.5 (LY04) | −0.4 % en el flujo sensible |
| L_v | 2.5e6 (verificado) | 2.5e6 | igual |
| ρ_aire | gas ideal con T_v [verificar] | gas ideal con T_v 0.608 | similar |
| SST en el bulk | temperatura potencial (promedio 1 m) | SST dado por el llamador | alinear con sec. 1 |
| Signos | sens=−t_flux, evap=−q_flux (MOM) | positivo hacia el océano | mismo convenio final |
| Albedo | opción 2, fuera de MOM6 [verificar] | `radiative_fluxes`: 0.066 o `albedo_large_yeager` | revisar valor FMS |
| LW | neta del hielo [verificar emisividad] | ε(LW↓−σT⁴), ε=0.97 | verificar ε FMS |

Conclusión: `bulk_ncar` reproduce la misma familia (LY04/CESM); las diferencias conocidas son `cp_air`, `q_sat`, alturas/ajuste 2->10 m,
gustiness y viento mínimo. No se puede afirmar igualdad bit a bit sin `surface_flux.F90`. `bulk_kara` (HYCOM flxflg=6) es otro
algoritmo y no aplica a MOM6.

## 8. Implicaciones para N2b

1. Implementar el camino MOM6 como tres bloques separados (como MOM6): (a) bulk NCAR a 10 m -> (stress, sens, evap); (b) conversión (`sens`, `latent = evap·L_v`, `lprec/fprec/lrunoff`);
   (c) aplicación en el océano (calor neto, SW penetrante por bandas, agua real que entra a T_top con S=0).
2. SST/SSS al bulk: promedio de 1 m de espesor (no la capa superior ni el valor del nivel 1) y velocidad = capa 1.
3. Forzamiento ERA5: desacumular antes, no usar `factor` para eso; interpolación lineal en el tiempo entre instantes etiquetados.
4. Penetración SW: MANIZZA_05, 3 bandas, 57 % vis / 43 % NIR, `BLUE_FRAC_SW = 0.5`; el albedo se debe aplicar antes.
5. Restoring de S: opcional, como sal flux, con tope ±5 y balance neto cero; OFF por defecto para regionales con OBC.

## [verificar] resumen
- Todo `ncar_ocean_fluxes` (n_itts, ψ, clips, viento mínimo, gust, T potencial, ρ_aire): fuente `FMScoupler/full/surface_flux.F90`.
- `ocean_albedo_option = 2` (esquema y valor), `ocean_rough_nml` Beljaars y `monin_obukhov_nml neutral` (si afectan a `ncar_ocean_flux`).
- Emisividad y fórmula del LW saliente en SIS2; promedio temporal de SST/u_surf en el paso de acoplamiento.
- Uso de `dhdt, dedt, drdt` (flujo implícito) en data_table.
- `gust = 1e-4` de data_table: cómo entra en el viento efectivo.
- Posible uso inconsistente de `depth(i)` vs `depth(I)` en `MOM.F90:4345` con `HMIX_UV_SFC_PROP > 0`.
- Si `useRiverHeatContent` se activa en OM4 y el tratamiento exacto del calor del runoff.
