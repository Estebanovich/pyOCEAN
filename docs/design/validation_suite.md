# Suite de validación cruzada: MITgcm · CROCO · MOM6 · HYCOM · pyOCEAN

Documento de diseño. Fecha: 2026-10-10.

## Objetivo

Correr los mismos casos de prueba en varios modelos de referencia y luego reproducirlos en pyOCEAN, comparando contra soluciones analíticas cuando existen.

## Rutas abreviadas

- **MIT** = `~/pyOCEAN_refs/MITgcm/verification/`
- **CROCO** = `~/pyOCEAN_refs/croco/TEST_CASES/croco.in.<X>`. La llave cpp está en `OCEAN/cppdefs.h`.
- **M6ex** = `github.com/NOAA-GFDL/MOM6-examples/ocean_only/<caso>` (no clonado).
- **M6src** = `~/pyOCEAN_refs/MOM6/src/user/<nombre>_initialization.F90`. Solo código: hay que escribir el `MOM_input`.
- **HYCOM:** solo hay casos realistas (GOMb0.08, GLBt0.72). **Ningún caso idealizado está listo.**

## 1. Matriz de casos

"Listo" significa que la configuración existe y fue verificada. "Propio" significa que hay que armarla a partir de una plantilla.

| Caso | MITgcm | CROCO | MOM6 | HYCOM | Referencia | Costo | Fase |
|---|---|---|---|---|---|---|---|
| Lock exchange | Propio (plantilla `internal_wave` 2-D) | **Listo** `GRAV_ADJ` (128×1×40) | **Listo** `lock_exchange` (128×4×21) | Propio (fase 3) | Frente U = ½√(g′H) (Benjamin 1968); deriva de RPE (Ilicak et al. 2012) | s–min | 1, 3 |
| Monte submarino (error del gradiente de presión) | Propio (`exp4`) | **Listo** `SEAMOUNT` | **Listo** `seamount/{z,sigma,rho,layer}` | Propio | Debe quedar en reposo; \|u\| espuria máxima (Beckmann & Haidvogel 1993) | min | 1, **3** |
| Marea interna sobre talud o cordillera | **Listo** `internal_wave` (60×1×20, OBCS) | **Listo** `INTERNAL`, `IGW` (marea en OBC) | — | — | Pendiente del haz √((ω²−f²)/(N²−ω²)); conversión (Llewellyn Smith & Young 2003) | min–1 h | 2, **4** |
| Seiche barotrópico | Propio | Propio (`TANK` es no hidrostático) | **Listo** `external_gwave` | — | Merian T = 2L/(n√(gH)) | s | 1, 2 |
| Seiche interno / sloshing | Reducción 2-D de `BayIW_Rectan_2layer` | — | **Listo** `sloshing/{z,rho,layer}` | — | Merian de dos capas; c = NH/(nπ) | min | 1, 3, 4 |
| Onda de Kelvin / atrapada a la costa | Propio | Propio | Solo código: `Kelvin`, `shelfwave` | — | c = √(gH); decaimiento a escala Rd; reflexión en OBC | min | 2 |
| Bahía con marea y OBC | Propio (OBCS de `internal_wave`) | Infraestructura de `IGW` | Solo código: `tidal_bay` | — | Amplitud en la bahía (Taylor); transparencia de la OBC | min | 2 |
| Doble giro barotrópico | **Listo** `tutorial_barotropic_gyre` | **Listo** `BASIN` | **Listo** `double_gyre` | — | Transporte de Sverdrup; capa de Munk (A_h/β)^(1/3) | min | 1 |
| Doble giro baroclínico | **Listo** `tutorial_baroclinic_gyre` | `BASIN` 3-D | `double_gyre` | — | Comparación entre modelos | ~1 h | 2 |
| Surgencia en plataforma | Propio | **Listo** `UPWELLING`, `INNERSHELF` | — | — | Transporte de Ekman τ/(ρ₀f) | min | 2 |
| Cañón submarino | Propio | **Listo** `CANYON` (65×48×16) | — | — | Escalamientos de Allen & Hickey 2010 | min | 2, 4 |
| Flujo pendiente abajo | **Listo** `dome` | **Listo** `OVERFLOW` | **Listo** `flow_downslope/{z,sigma,rho,layer}`, `DOME` | Propio | Descenso y entrainment de la pluma (Legg et al. 2006) | min–1 h | **3** |
| Inestabilidad baroclínica | **Listo** `tutorial_reentrant_channel` | **Listo** `JET` | `Phillips_2layer` | — | Tasa de crecimiento de Eady/Phillips | horas | 3, 4 |
| Advección de trazadores | **Listo** `advect_xy`, `advect_xz` | **Listo** `TS_HADV_TEST` | **Listo** `torus_advection_test` | — | Exacta: errores L1, L2, L∞; extremos; orden de convergencia | s | 1 |
| Columna 1-D con KPP | **Listo** `vermix` | **Listo** `SINGLE_COLUMN` (con `LMD_MIXING`) | **Listo** `CVmix_SCM_tests` | — | Kato–Phillips, h ≈ 1.05 u*(t/N₀)^½ (Price 1979); convección | s | 1 |
| Columna de flujos aire–mar | **Listo** `1D_ocean_ice_column` (EXF + KPP) | Propio | **Listo** `single_column` (BATS) | Vía GOMb0.08 | Q_net, τ, SST(t) | s | 2 |

**No aplican**, porque son no hidrostáticos: CROCO `TANK`, `ACOUSTIC` y `KH_INST`; MIT `short_surf_wave`.

## 2. Lista priorizada para pyOCEAN

1. **Lock exchange**
   - Modelos: CROCO `GRAV_ADJ`, MOM6 `lock_exchange` y MITgcm propio.
   - Métricas: x_f(t) contra ½√(g′H); RPE(t) con upwind y con PPM.
2. **Advección de trazadores**
   - Modelos: CROCO `TS_HADV_TEST`, MOM6 `torus_advection_test`, MIT `advect_xy`.
   - Métricas: errores L1, L2 y L∞; preservación de extremos; orden de convergencia.
3. **Columnas KPP**
   - Modelos: CROCO `SINGLE_COLUMN`, MOM6 `CVmix_SCM_tests`, MIT `vermix`.
   - Métricas: h(t) contra Price (1979) y Deardorff; perfiles T(z).
4. **Doble giro barotrópico**
   - Modelos: MIT, CROCO `BASIN` y MOM6.
   - Métricas: Sverdrup y ancho de la capa de Munk.
5. **Seiches, barotrópico e interno**
   - Modelos: MOM6 `external_gwave` y `sloshing`; reducción 2-D de BayIW en MITgcm; CROCO propio.
   - Métricas: periodos contra Merian; decaimiento. Valida la superficie libre implícita.
6. **Columna de flujos aire–mar**
   - Modelos: MIT `1D_ocean_ice_column` y MOM6 `single_column`.
   - Complementa `validation/mitgcm_exf` y `validation/hycom_kara`.
7. **Onda de Kelvin y bahía con marea con OBC**
   - Modelos: MITgcm propio, CROCO y MOM6 (solo código).
   - Métricas: c = √(gH) y escala Rd; energía reflejada menor a 5 %.
8. **Marea interna sobre talud o cordillera**
   - Modelos: MIT `internal_wave`, CROCO `INTERNAL` e `IGW`.
   - Métricas: ángulo del haz, conversión de M2 y amplitudes modales. Es el caso más cercano a la tesis.
9. **Monte submarino y flujo pendiente abajo**
   - Modelos: MOM6 (z, sigma, rho y capas), CROCO, MIT `dome` y HYCOM propio.
   - Sirve de criterio de entrada a la Fase 3.

Casos extra, de un solo modelo: CROCO `CANYON` y `UPWELLING` (Fase 4) y MIT `tutorial_reentrant_channel`.

## 3. Notas prácticas (macOS arm64)

Ya instalado en Homebrew: gfortran, mpich, netcdf, netcdf-fortran (`nf-config`) y hdf5.

**MITgcm** (la más fácil, ~0.5 día)
- Comandos: `genmake2 -mods ../code -of tools/build_options/darwin_arm64_gfortran`, luego `make depend && make`.
- Sin MPI ni netCDF; la salida es MDS y se lee con `xmitgcm` o `MITgcmutils`.
- Los `data` de verification corren solo 10–100 pasos; hay que subir `nTimeSteps`.

**CROCO** (~0.5–1 día para el primer build)
- `create_config.bash`, luego en `cppdefs.h` quitar `REGIONAL` y definir el caso, luego `./jobcomp`.
- Los casos de prueba son seriales.
- Si el cpp de clang rompe el Fortran, usar el cpp de GCC de Homebrew.

**MOM6** (la más difícil, ~1–2 días)
- Necesita FMS, MPI y netCDF-Fortran. La ruta más simple es el build de `.testing`, que clona FMS (requiere red).
- MOM6-examples se clona sin `--recursive`.

**HYCOM** (no realista para casos idealizados: 2–4 días por caso)
- Cada caso requiere `regional.grid/depth .a/.b`, `blkdat` y archivos de relajación hechos a mano.
- Solo vale la pena en la Fase 3: monte submarino, flujo pendiente abajo y lock exchange.

**Cobertura sugerida**
- MITgcm + CROCO como par principal para los casos 1–8.
- MOM6 donde sea la única configuración lista, incluidas las familias z/sigma/rho/capas.
- HYCOM solo en la Fase 3; mientras tanto, comparar con las corridas operacionales (PAC_MEX, GoM).

## 4. Estado

| Validación | Estado |
|---|---|
| `validation/hycom_kara`: `bulk_kara` contra el código compilado de HYCOM | ✅ Coincide a 1e-10 (616 casos) |
| `validation/mitgcm_exf`: `bulk_ncar` contra MITgcm `exf` (LY04 y LY09, 1000 casos) | ✅ Mediana 0.2–0.3 % para U ≥ 0.5 m/s. Todas las diferencias mayores tienen causa documentada: c_pa 1005 vs 1000.5; MITgcm usa la velocidad sin recortar con U < 0.5; 2 iteraciones no convergen con viento débil y estable. Para reproducir: `./build_and_run.sh all && python compare.py all` |
| `validation/bayiw_seiche`: seiche barotrópico de la bahía BayIW contra las corridas de MITgcm de la tesis | ✅ Periodo 3.83 h contra 3.81–3.83 h; correlación 0.990–0.998 y RMSE ≤ 0.5 mm en 0–27 h |
| Casos 1–9 | Pendientes |
