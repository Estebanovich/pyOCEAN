# Recorte de NWA12: caso regional real (cadena regional, Tareas 9–11)

- **Spec:** `docs/superpowers/specs/2026-11-01-regional-chain-design.md` (criterios 4, 6 y 8; "Decisiones del caso")
- **Plan:** `docs/superpowers/plans/2026-11-01-regional-chain.md`, Tareas 9 (caso), 10 (MOM6) y 11 (comparación)
- **Estado (2026-11-03):** 🔶 — criterios 1–5 y 7 ✅, 6 ✅ salvo el costo (4× MOM6), 8 ✅; resumen al final ("Tarea 11")
- **Lecturas de referencia:** `docs/design/regional_refs/` (malla, OBC y mareas, presión y ríos, herramientas de CEFI)

## Qué es

El Mid-Atlantic Bight y el sur de Nueva Inglaterra en la malla de NWA12 de CEFI, del 2017-05-01 al 2017-09-01, con
las fronteras abiertas de GLORYS12, la marea de TPXO10, ERA5 por el bulk y los ríos de GloFAS. Es el primer caso
regional real de pyOCEAN de punta a punta; el mismo caso lo corre MOM6 (Tarea 10).

Todo lo general está en la biblioteca (malla de supergrid, GLORYS, mareas 3D, ríos, presupuesto del dominio); aquí
solo vive lo propio del caso:

| archivo | qué hace |
|---|---|
| `case.py` | ventana, periodo, rutas, reglas de profundidad de MOM6, `parity_config()`, `build()` |
| `build_inputs.py` | escribe la malla, la vgrid, los ríos y las estaciones a `RUNS/inputs/pyocean/` |
| `run_pyocean.py` | corridas `tides` y `full`, chequeo de restart y de hilos, presupuestos, costo |
| `obs_qc.py` | control de calidad de las observaciones (mareógrafos CO-OPS, boya NDBC 44025) antes de cualquier número |
| `compare.py` | modelos contra observaciones (Tarea 11): mareas, η, SST contra GLORYS, OISST y 44025 |
| `tide_check.py` | vistazo armónico de una corrida en los mareógrafos (lee la SSH de MOM6 como promedio de la hora de acople) |

Datos y salidas en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/` (`data/`, `inputs/`, `runs/`), nunca en el repo.

## Ventana

`choose_window()` sobre `ocean_hgrid.nc` y `ocean_topog.nc`, con las reglas del spec y la decisión R13:

1. la caja de índices de los puntos T dentro de 75.5–68.5 °W y 36.5–42.0 °N (R13: el límite norte sube de 41.8 a
   42.0 °N para que entren Narragansett/Blackstone y Buzzards Bay);
2. cada borde con alguna celda mojada es abierto, y cada borde abierto queda a ≥ 10 celdas de NDBC 44025
   (40.251 °N, 73.164 °W) y de OOI Pioneer (40.1 °N, 70.8 °W); un borde más cerca se mueve hacia afuera;
3. ningún borde abierto tiene celdas con H ≤ 0.

Resultado: **i = 282…368, j = 425…513 (`window = (282, 369, 425, 514)`), 87 × 89 columnas × 75 niveles, 6199
columnas mojadas** (6201 antes de la edición R14), H de 5 a 4811 m. La malla es lon/lat regular en este recorte (ángulo cero). Distancias a los
bordes: 44025 a 28 celdas del oeste y 29 del norte; Pioneer a 28 del este y 31 del norte.

**Edición de la topografía del caso (decisión R14, clase c).** Una frontera abierta de GLORYS a 1/12° no puede
cortar el interior de un estuario. `edit_edges()` pone como tierra cada tramo de menos de 4 celdas mojadas
contiguas sobre un borde de la ventana. En el recorte eso cierra **2 celdas de la parte alta de la bahía de
Delaware en el borde oeste**: (j = 43, i = 0) en 39.236 °N, 75.440 °W y (j = 44, i = 0) en 39.298 °N, 75.440 °W
(índices del recorte; i = 282, j = 468–469 de la malla completa). El río Delaware entra como descarga de GloFAS. La
profundidad editada **es la topografía del caso** para pyOCEAN y para MOM6: `build_inputs.py` escribe
`ocean_topog_cut.nc` y `ocean_mask_cut.nc`, y la Tarea 10 los usa. Quedan 6199 columnas mojadas; la bahía de
Delaware sigue unida al mar por su boca.

Los **cuatro bordes siguen abiertos** por la regla 2:
- sur (87 celdas mojadas de 87) y este (89 de 89): océano abierto;
- oeste, 23 celdas: el Atlántico frente a la península de Delmarva, al sur de 37.9 °N;
- norte, 25 celdas en dos tramos: la bahía de Cape Cod (6 celdas, 70.56–70.16 °W) y el golfo de Maine al este de
  70.0 °W (19). El tramo de Cape Cod no está unido al resto dentro de la ventana (24 celdas, 41.79–41.97 °N), pero
  intercambia con la bahía de Massachusetts fuera del recorte: con 6 celdas queda abierto (cerrarlo lo volvería una
  cuenca cerrada). Decisión R18: la regla es solo la longitud del tramo, no la conectividad dentro de la ventana.

Profundidades como MOM6 (`MASKING_DEPTH` 1, `MINIMUM_DEPTH` 4, `MAXIMUM_DEPTH` 6500): tierra donde la profundidad es
≤ 1 m, las de 1 a 4 m suben a 4 m.

Estaciones (`stations()`): los 13 mareógrafos de CO-OPS bajados (`data/tides/`), 44025 y Pioneer, cada uno en su
celda T mojada más cercana. Providence (8454000) y Kings Point (8516945) no tienen celda mojada a menos de 3 celdas
y no se registran.

## Configuración de paridad con NWA12

`parity_config()`; los valores vienen de `MOM_parameter_doc.short` de NWA12.COBALT y de las decisiones R3–R5, R7,
R11 y R12 del registro.

| tema | NWA12 (MOM6) | pyOCEAN | diferencia |
|---|---|---|---|
| vertical | z\*, 75 niveles de `vgrid_75_2m.nc` | ALE z\*, `dz_top` = los mismos 75 Δz | — |
| momento | vector-invariante; Smagorinsky biarmónica `SMAG_BI_CONST` 0.03, `AH_VEL_SCALE` 0.01, `BOUND_AH`, `OBC_ZERO_BIHARMONIC` | vector-invariante; Smagorinsky **biarmónica** `smagorinsky_bi` 0.03, `ah_vel_scale` 0.01, laplaciana apagada (desde `1789e9f`, 2026-11-03; antes laplaciana 0.15 de OM4) | (c) cota × 6/11 para AB3 |
| gravedad | `G_EARTH` 9.8 | `g=9.8` (R21): llega a la superficie libre y Flather, al PGF, a la marea de potencial/SAL, a los cierres y a la presión de la EOS; el bulk de FMS usa su propio GRAV 9.80 | — |
| peso de masa en el PGF | `MASS_WEIGHT_IN_PRESSURE_GRADIENT` True | `pgf_mass_weight='full'` (R21; PGF inicial = MOM6 a 1.2e-14, Tarea 10) | — |
| EOS | `WRIGHT_FULL` (θ, S_P), ρ₀ 1035, `C_P` 3992 | `eos='wright_full'`, `ts_kind='theta_sp'`, ρ₀ 1035, c_p 3992 | — |
| superficie | bulk NCAR del acoplador FMS, con SIS2 | `BulkFluxes('ncar')` (= `surface_flux` de FMS bit a bit), ERA5 con `lsm` | (c) sin SIS2, sin ráfagas |
| onda corta | `VAR_PEN_SW`, `OPACITY_SCHEME` MANIZZA_05 con clorofila de archivo, 3 bandas (`mom6_full`; `mom6_full_jerlov` es el control con Jerlov) | Jerlov I, Paulson & Simpson | (c) |
| mezcla vertical | ePBL + Jackson + doble difusión, `KD` 1e-6, `KV` 0 | KPP (CVMix) + Jackson, κ_bg 1e-6 (el `KD` de NWA12, pasado como parámetro; el valor por defecto de pyOCEAN sigue en 0), ν_bg 0 | (b) sin ePBL ni doble difusión |
| mezcla por cizalla | Jackson, `PRANDTL_TURB` 1.25 (ν = 1.25 κ; `MOM_kappa_shear.F90:393, 773`) | `shear_mixing='jackson'`, `shear_prandtl=1.25` | — |
| Coriolis | `BOUND_CORIOLIS = False`, `CORIOLIS_EN_DIS = True` (de dos transportes elige el que disipa energía; `MOM_CoriolisAdv.F90:439-449, 733-815`) | flujo de PV de Sadourny con el acotado de BOUND_CORIOLIS siempre activo (`core/components/momentum.py:210`, no es parámetro), sin `CORIOLIS_EN_DIS` | (c) |
| reestratificación de la capa de mezcla | `MIXEDLAYER_RESTRAT = True` (Fox-Kemper et al. 2011), coeficiente 1.0, `MLE_FRONT_LENGTH` 1500 m, MLD de ePBL | no existe | (b), encontrada el 2026-11-04 |
| esquema temporal | RK2 dividido, `BE` 0.6; barotrópico split-explicit (`BEBT` 0.2, `DTBT` −0.9) | AB3; superficie libre implícita θ = 0.6, θ_bc = ½ | (c) |
| trazadores | `PPM:H3` con `DT_THERM` = `DT_TRACER_ADVECT` = 1800 s | WENO5-Z + FCT en SSP-RK3 (tres etapas) cada 600 s, un solo Δt | (c) |
| fricción de fondo | cuadrática `CDRAG` 0.003, `HBBL` 10 m, `DRAG_BG_VEL` 0, `CHANNEL_DRAG` | cuadrática cd 0.003, hbbl 10, U_bg 0 | (b/c) sin `CHANNEL_DRAG` (R5) |
| bordes abiertos | Flather + Orlanski + nudging 3 d / 360 d (según la velocidad de fase de Orlanski), reservorio de trazadores de 9 km | `'radiating'`: Flather + Orlanski + nudging τ_in 3 d / τ_out 360 d (según el signo de la velocidad de fase de Orlanski, como CROCO, `core/components/open_boundary.py:150`; R7 decía «según el sentido del flujo», corregido el 2026-11-04), sin reservorio ni esponja | (c) |
| velocidad tangencial en los bordes abiertos | `ORLANSKI_TAN`, `NUDGED_TAN`: radiada y relajada hacia la velocidad tangencial exterior | `obc_tangential='mom6'` (R27, desde `2f80d5a`): la misma radiación y relajación, ζ y D_S calculadas en las esquinas del borde (antes ζ = D_S = 0, el ZERO de MOM6) | — |
| exterior | GLORYS diario en sus niveles (`z_to_dz` de CEFI), remapeado con PPM_H4 | GLORYS diario en sus niveles (`source_level_grid`, interfaces a medio camino), remapeado a las capas, lineal en el tiempo; ū = media de la columna de la celda de borde | — |
| marea de borde | TPXO, 10 constituyentes, η y la velocidad de TPXO sumadas al segmento, nodal fijo en una fecha | `BoundaryTides` con TPXO10 (10 constituyentes), `velocity_depth='tpxo'`: u = U/h_TPXO, la velocidad propia de TPXO (R19, revisa R3 para el caso; u = U/D del modelo daba hasta 3.8 m/s en caras de 5 m), nodal OTPS en cada instante, sin rampa | (c) nodal |
| marea de potencial | `TIDES`, SAL escalar 0.05, SAL en Flather | `tidal_potential=True`, `sal_beta` 0.05, `sal_flather` | (c) camino SAL antiguo de MOM6 (R4) |
| presión atmosférica | apagada (`MAX_P_SURF` = 0) | `atm_pressure=False` | — |
| ríos | GloFAS v3 con la regla de CEFI; mezcla de ríos de ePBL (`DO_RIVERMIX`) | GloFAS v4 con la regla `ldd` de CEFI (R11, R12), agua dulce real en la capa superior; desembocaduras elegidas (caudal medio ≥ 1 m³/s) sobre todo el periodo del caso, 72 celdas, igual para toda corrida y todo restart (R17) | (c) versión; (b) sin mezcla de ríos |
| Δt | `DT` 600 s, `DT_THERM` 1800 s | un solo Δt de 600 s (300 s hasta la corrección del flujo de PV, `e1ffbbb`) | (c) |
| estado inicial | T, S, SSH y velocidades de GLORYS, remapeo desde z | T, S remapeados desde los niveles de GLORYS (contenido exacto por columna), η = zos (sin quitar la media, como MOM6), u, v rotados e interpolados linealmente en la vertical a cada cara | (c) velocidades interpoladas, no remapeadas |

**Por qué ALE z\* + momento vector-invariante + Smagorinsky.** El momento en forma de flujo de σ y la laplaciana de
A_h constante no tienen los términos métricos curvilíneos (Tarea 3, `docs/design/regional_refs/metric_audit.md`
§2.3, clase b). Las formas invariantes sí son correctas en una malla curvilínea, como en MOM6. `parity_config()`
rechaza otra combinación, y el pre-flight avisa (`PyOceanConfigWarning`) si una malla de supergrid se usa con la
forma de flujo o con la laplaciana constante.

**Decisiones aceptadas (R15, R16).** Δt = 300 s para la corrida completa: era la inestabilidad conocida de
θ_bc = ½ (clase a; MOM6 corre con 600 s), no un ajuste de física. **Corregida el 2026-11-02 (`e1ffbbb`):** el
flujo de PV tomaba el transporte del paso anterior; desde entonces la paridad corre con 600 s. κ_bg = 1e-6 es el valor de
paridad con NWA12; el punto (2) de "qué debe mejorar" de N2b prefiere decidir la mezcla de fondo contra la boya
44025, así que se revisa en el proyecto 4. Sin banda de esponja (NWA12 no tiene) y velocidades iniciales
interpoladas en la vertical (MOM6 las remapea): aceptadas, clase c.

**Física de NWA12 que pyOCEAN no tiene (clase b o c, para el proyecto 4):** ePBL, doble difusión, la reestratificación
de la capa de mezcla de Fox-Kemper (`MIXEDLAYER_RESTRAT`), la onda corta de Manizza con clorofila, `CHANNEL_DRAG`, el
reservorio de trazadores de los segmentos (9 km), la mezcla de ríos de ePBL y SIS2; además, elecciones numéricas (c):
`CORIOLIS_EN_DIS`, RK2 con `BE` 0.6 + barotrópico split-explicit, trazadores `PPM:H3` con `DT_THERM` 1800 s. MOM6 corre
con `ENABLE_BUGS_BY_DEFAULT = True` (banderas de bugs de OBC activas). El artículo de Ross et al. (2023) da Smagorinsky
0.015 y SAL 0.01; el `MOM_input` de CEFI y de `mom6_full` tienen 0.03 y 0.05: la paridad sigue a `MOM_input`.
Detalle: `reports/Deriva norte de la Corriente del Golfo.md` (sin commit) y `figs/gs_drift/`. (La Smagorinsky biarmónica y la velocidad tangencial de MOM6 en los bordes ya están.)
**MOM6 "como pyOCEAN" (d11 fase 3, 2026-11-04; `run_mom6.sh full_aspy`, `mom6_inputs.ASPY_SWITCHES`):** apagar a la
vez MLE, `CORIOLIS_EN_DIS` (con `BOUND_CORIOLIS`), `DT_THERM` 1800 → 600, el reservorio de trazadores y la doble
difusión no acerca la trayectoria de MOM6 a la de pyOCEAN: mediana de la pared norte a 72 °W (días 60–123) 38.17 y
37.60 °N en dos miembros (MOM6 37.83 / 37.75, pyOCEAN 38.93 / 38.90); lectura pre-registrada AMBIGUA → SIN EFECTO en la
confirmación, sin bisección. La causa queda en lo estructural (ePBL vs KPP, RK2/BE vs AB3, barotrópico) [hipótesis].
La dispersión intrínseca de MOM6 es ≥ 0.57° (no 0.08°): toda comparación de trayectoria necesita ≥ 2–3 miembros.

## Cómo reproducirlo

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=5
python validation/nwa12_cut/build_inputs.py                                   # entradas → RUNS/inputs/pyocean/
python validation/nwa12_cut/run_pyocean.py --mode tides --days 30 --dt 600    # mareas solas (criterio 4)
python validation/nwa12_cut/run_pyocean.py --mode full --days 30 --dt 600 --restart-at 15   # corridas de la Tarea 9
python validation/nwa12_cut/run_pyocean.py --mode full --days 30 --dt 600 --start-from \
    /Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/runs/full_dt600/restart_day15.nc --tag from_restart
python validation/nwa12_cut/run_pyocean.py --mode full --compare \
    /Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/runs/full_dt600/final.nc \
    /Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/runs/full_dt600_from_restart/final.nc
python validation/nwa12_cut/run_pyocean.py --mode full --days 30 --dt 300
python validation/nwa12_cut/run_pyocean.py --mode full --threads-check 3 --dt 600   # 1 contra 5 hilos, 3 horas
```

**La corrida de paridad de 123 días**, cuando ERA5 esté completo hasta el 2017-09-01 (el modo `full` se niega a
arrancar si ERA5 no cubre la corrida y dice hasta cuándo cubre):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=5
python validation/nwa12_cut/run_pyocean.py --mode full --days 123 --dt 300 --threads 5 --restart-at 60.5
```

Δt = 300 s era el estable medido antes de `e1ffbbb` (abajo); tarda ≈ 6.8 h. **La paridad vigente** (biarmónica,
`obc_tangential='mom6'`, Δt 600 s, 123 días en ≈ 3.6 h): `run_pyocean.py --mode full --days 123 --dt 600 --threads 5
--tag bih123`; con salida horaria y 30 días, `--days 30 --tag bih30h` (la usa `compare.py` para medir el sesgo de las
instantáneas). Para el chequeo del restart a la mitad:
`--start-from …/runs/full_dt300/restart_day60.5.nc --tag from_restart` y luego `--compare` de los dos `final.nc`.
El restart cae a las 12:00, entre dos registros diarios de GLORYS y GloFAS y entre dos horas de ERA5, y se escribe con
`restart_day60.5_budget.npz`, los acumuladores del presupuesto, que `--start-from` recarga (si no existen, el
presupuesto de la continuación empieza en el restart y `meta.json` lo dice en `budget_scope`).

Cada corrida escribe en `RUNS/runs/<modo>_dt<Δt>[_<tag>]/`: `out/out_<fecha>.nc` (NetCDF4 CF, un archivo por
registro: η, T, S, u, v y el grosor de capa, cada 24 h por defecto; un archivo por registro porque la consolidación
de metadatos de Zarr falla con los archivos `._*` que macOS escribe en el disco externo), `series.npz` (η, SST y
SSS en las estaciones, cada paso), `budget.npz` (modo `full`), `final.nc` (restart del final), `meta.json`
(truncamientos, eventos del limitador, costo por día simulado, presupuestos).

## Presupuestos del dominio

`diagnostics/domain_budget.py` (`DomainBudget`), en cada paso:

- volumen: ΔV = Δt (Q_abierto + Σ A fw), con fw = P − E del bulk más los ríos;
- calor: Δ(VT) = Δt F_abierto^T + Δt Σ A (Q_ns + Q_sw)/(ρ₀ c_p) + Σ (V_fw − V_adv) T (el agua dulce entra con la T de
  la capa superior);
- sal: Δ(VS) = Δt F_abierto^S.

F_abierto es el flujo efectivo del FCT en las caras abiertas (F_L + C·A, Zalesak 1979), que el paso de advección
registra con `TracerAdvection.record_open_flux`; el resto de los operadores (remapeo, difusión vertical, término no
local de KPP) conserva el contenido. El
residuo acumulado se mide contra los términos del propio presupuesto, max(|Δ|, max |término acumulado|)
(`rel_terms`, la medida estricta), contra el flujo bruto Σ|términos| (`rel_gross`, que el flujo de marea por las
caras abiertas infla) y contra el contenido.

Análisis armónico en los mareógrafos: `python validation/nwa12_cut/tide_check.py RUN_DIR … [--boundary]`.

## Resultados (2026-11-01; Tarea 9)

Todas en 5 núcleos P (`NUMBA_NUM_THREADS=5`), 87 × 89 × 75.

| corrida | Δt | días | truncamientos CFL | eventos del limitador | puntos limitados de Smagorinsky | costo |
|---|---|---|---|---|---|---|
| mareas solas (R19 + R21) | 600 s | 30 | 0 | 0 | 25 885 (≈ 6 por paso) | 0.71 s/paso, 102 s por día simulado |
| completa (paridad) | 600 s | 30 (mayo; ERA5 cubría solo mayo) | 0 | 0 | 25 218 | 0.736 s/paso, 106 s/día |
| completa (paridad) | 300 s | 30 | 0 | 0 | 0 | 0.689 s/paso, 198 s/día |
| **completa (paridad, R19 + R21), 2017-05-01 → 09-01** | 300 s | 123 | 0 | 0 | 0 | 0.706 s/paso, **203 s/día** (6.9 h) |

**Δt estable más grande: 300 s para la corrida completa; 600 s para mareas solas.** Con 600 s la corrida completa
termina sin truncamientos, pero desde el día ~25 crece un modo de la capa superior en la zona de la Corriente del
Golfo y los anillos (37.8–38.9 °N, 70.7–69.4 °W, lejos de los bordes): oscilación inercial de los ~300 m de arriba
que pasa de |u|máx ≈ 2 m/s (GLORYS) a 4.2 m/s en el día 28 (percentil 99 de la capa superior 1.5 → 3.3 m/s, 746
caras con |u| > 2 m/s). Es la inestabilidad conocida del transporte baroclínico centrado en el tiempo
(`theta_bc = ½`; fila DOME de CLAUDE.md, allí a fΔt = 0.12, aquí a fΔt ≈ 0.054): **diagnóstico** (no cambio de
configuración) — la misma corrida desde el restart del día 15 con θ_bc = 1 se queda en |u|máx 1.84 m/s y p99 1.57
el día 30, igual que la de 300 s (1.73 y 1.50). Clase (a) del núcleo, con su plan aparte; no se corrigió aquí. Con
300 s no aparece: |u|máx de la capa superior ≤ 1.95 m/s en los 30 días.

**Costo de los 123 días:** a 300 s, 123 × 198 s ≈ 6.8 h; a 600 s, 3.6 h (pero con el modo inestable). La aceleración
de 1 a 5 hilos es solo 1.6× (1.04 → 0.64 s/paso a 600 s): el paso tiene mucha parte en serie.

**Presupuestos del dominio — criterio 6 ✅** (R20: "≤ 1e-10 relativo" se lee contra el contenido del dominio y
contra el flujo bruto, es decir, cierre a redondeo). Residuo acumulado relativo al contenido / al flujo bruto
Σ|términos| (`rel_gross`) / al término más grande del propio presupuesto, max(|Δ|, max |término acumulado|)
(`rel_terms`, la medida más estricta, informativa):

| corrida | volumen | calor | sal |
|---|---|---|---|
| completa, 600 s, 30 días | 4.5e-16 / 8.0e-15 / 2.7e-12 | 1.0e-14 / 8.5e-14 / 2.5e-12 | 1.7e-14 / 3.0e-13 / 3.2e-11 |
| desde el restart del día 15, 600 s | 2.6e-16 / 8.7e-15 / 1.5e-12 | 5.3e-15 / 7.3e-14 / 2.1e-13 | 8.6e-15 / 2.9e-13 / 1.7e-10 |
| completa, 300 s, 30 días | 6.6e-16 / 1.2e-14 / 5.0e-12 | 2.0e-14 / 2.2e-13 / 1.1e-12 | 5.8e-14 / 1.0e-12 / 8.9e-11 |
| completa, 300 s, 10 días (R17) | 3.5e-16 / 1.9e-14 / 1.0e-12 | 5.9e-15 / 1.6e-13 / 2.0e-13 | 1.8e-14 / 9.9e-13 / 1.1e-10 |
| **completa, 300 s, 123 días (R21)** | 3.8e-16 / 1.7e-15 / 1.6e-12 | 9.8e-14 / 1.1e-13 / 2.9e-12 | 2.2e-13 / 9.8e-13 / 4.4e-10 |

Contra el contenido y el flujo bruto todo cierra a ≤ 1e-12. Con la medida más estricta la sal llega a 1.1–1.7e-10
en los tramos de 10–15 días: el flujo neto de sal es chico (3e12 g kg⁻¹ m³ en 10 días) y el residuo, ~150–1100
g kg⁻¹ m³ (≈ 2e-14 del contenido de sal), es redondeo de la aritmética del modelo. Crece con el tamaño del campo
(calor ~17, sal ~350 en los mismos 10 días), no con los flujos, y no viene de las sumas del dominio (llevar el
presupuesto como Σ V (S − S₀) no lo cambia: 339 → 350). No falta ningún término.

**Restart a mitad de la corrida de 123 días (criterio 6 ✅):** 123 días seguidos contra 60.5 días + restart (a las
12:00, entre registros diarios de GLORYS/GloFAS y entre horas de ERA5) + 62.5 días: las 19 variables del restart final
y todas las series por paso de las estaciones (η, SST, SSS, max |u|) son idénticas bit a bit, y los acumuladores del
presupuesto, recargados en el restart, dan el mismo residuo.

**El modo de θ_bc = ½ también aparece a 300 s** en la corrida de 123 días: crece desde el día ~40–45 (oscilación de
18.9 h en max |u|) y desde el día ~75 la capa superior de la Corriente del Golfo llega a 5.5–7.2 m/s (max |u| de la
corrida 7.25 m/s, sin truncamientos). Diagnóstico en `figs/gs_diag/README.md`; prueba con θ_bc = 1 en la sección 14
del informe de la Tarea 9.

**Restart y hilos:** 30 días seguidos contra 15 días + restart + 15 días (600 s, con GLORYS, mareas, ERA5 y ríos
variando): las 19 variables del restart final y todas las series por paso de las estaciones son idénticas bit a bit.
1 contra 5 hilos en 3 horas de la corrida completa: idénticos (η, u, v, T, S, h, estado del cierre, contadores).

**Mareas (vistazo, sin puntaje; el criterio 4 se juzga contra MOM6 en la Tarea 10):** corrida de mareas solas de 30
días a 600 s con la configuración de paridad (`velocity_depth='tpxo'` R19, g = 9.8 y `pgf_mass_weight='full'` R21;
tag `r21`): 0 truncamientos, 0 eventos del limitador, 102 s/día, |u|máx de la corrida 1.57 m/s (3.39 m/s con u = U/D
del modelo). M2 en 11 mareógrafos de
CO-OPS (`tide_check.py`, días 1–30, M2 S2 N2 K1 O1 Q1): **error vectorial medio 0.228 m, mediana 0.199 m**; TPXO10 en
los mismos puntos, 0.15 m (casi todo Nantucket y Woods Hole, que la malla de 1/12° no resuelve). Las amplitudes
están cerca (Bridgeport 0.92 contra 0.98 m, Atlantic City 0.63 contra 0.58), pero la fase de pyOCEAN se adelanta en
la bahía de Nueva York y Long Island Sound: **Sandy Hook −18.6°, Battery −31.3°, Montauk −27.0°**, Bridgeport −11.6°;
Atlantic City −4.0°, Lewes −4.6°. La velocidad de la marea en el borde no es la causa: con u = U/D del modelo (R3)
y u = U/h_TPXO las fases difieren ≤ 0.4° (16 días: error medio 0.185 contra 0.183 m). El adelanto es interior
(batimetría de 1/12° en NY Harbor–LIS, sin `CHANNEL_DRAG`; clase b/c); no se afina nada y se compara con MOM6 en el
mismo recorte.

## MOM6 + SIS2 en el mismo recorte (Tarea 10)

| archivo | qué hace |
|---|---|
| `mom6_inputs.py` | escribe las entradas de MOM6 en `RUNS/inputs/mom6/` desde la misma definición de caso; `--run-dir MODO` arma el directorio de una corrida |
| `build_mom6.sh` | compila MOM6 + SIS2 + acoplador FMS (`full`) + `atmos_null` + `land_null` en `~/pyOCEAN_refs/build/nwa12_cut/` |
| `run_mom6.sh {pgf0,tides,full,full_jerlov}` | corre en ≤ 5 rangos MPI y escribe `timing.json` |
| `check_mom6.py` | criterio 1 (malla), criterio 7 (PGF inicial), serie horaria de η en los mareógrafos para `tide_check.py`, costo |
| `MOM_input`, `SIS_input` | los de NWA12 (`exps/NWA12.COBALT/INPUT`) sin comentarios, valores tal cual |
| `MOM_override`, `SIS_override` | **todas** las diferencias con NWA12 (abajo); `run_mom6.sh` agrega el bloque del modo |
| `diag_table`, `data_table`, `data_table_tides`, `input.nml`, `field_table` | plantillas de la corrida |

**Compilación.** MOM6 `0e9a561` (GFDL; CEFI fija su fork `dev/cefi` af82618, que no está en local), SIS2 c287b24,
icebergs fd6478f, ice_param 1d19a57, atmos_null fc39f82, land_null 57e5942, FMScoupler 175a9da, todos con los
commits de los submódulos de CEFI. **FMS 16bb3af** (`~/pyOCEAN_refs/FMS_2025.03`, el commit de CEFI; `git describe`
dice 2026.01.01): el FMS local 2025.02.01 no puede compilar este acoplador (usa `fms_error_mesg` y
`fms_check_nml_error`, que `libFMS.F90` exporta desde 2025.03), así que se clonó al lado según la decisión del
controlador; el FMS 2025.02.01 no se tocó. Dependencias con `ac/makedep` de MOM6, memoria `dynamic_symmetric`, `-O2`
con reales de 8 bytes, sin COBALT (stubs `GFDL_ocean_BGC`), sin Icepack (stubs de SIS2), sin `use_yaml` (tablas de
texto). 3 min con 5 trabajos. No hizo falta FRE-NCtools (abajo).

**Entradas (`mom6_inputs.py`).** `xesmf` y `HCtFlood` no están instalados en ningún entorno, así que las
herramientas de CEFI no corren tal cual. Las funciones puras de CEFI (`rotate_uv`, `fill_missing`, `z_to_dz`,
`ap2ep`, `ep2ap`) se compilan desde `tools/boundary/boundary.py` mismo; los pasos de xesmf/HCtFlood/CDO se
reimplementan (clase c): `nearest_s2d` = punto fuente más cercano en distancia cartesiana sobre la esfera,
`cdo setmisstonn` = vecino válido más cercano en el mismo nivel, `flood_kara` = inundación iterativa con la media de
los 8 vecinos [verificar contra HCtFlood: su código no está en `~/pyOCEAN_refs`].

- malla: `ocean_hgrid.nc` cortado con desplazamientos pares; `ocean_topog.nc` = la topografía del caso (reglas de
  MOM6 + R14), float64; máscaras desde ella; `ocean_mosaic.nc`, `grid_spec.nc` de NWA12; las tres mallas de
  intercambio de NWA12 reindexadas a la ventana (atmósfera = tierra = océano, una celda de intercambio por celda; las
  2 celdas de R14 pasan de atm×océano a atm×tierra). Sin FRE-NCtools.
- **estado inicial = el de pyOCEAN** (`ic_full.nc`, `ic_tides.nc`: h, θ, S_P, u, v de `case.build`, superficie
  primero), leído capa por capa (`THICKNESS_CONFIG = "thickness_file"`, `REMAP_AFTER_INITIALIZATION = False`), así
  que los dos modelos arrancan del mismo archivo. `write_glorys_initial.py` de CEFI no se usa (daría otro estado).
- segmentos: 4 (NWA12 numera sur 1, norte 2, este 3; el oeste es el 4, `I=0,J=N:0`), GLORYS con el algoritmo de
  `write_glorys_boundary.py` (vecino más cercano en los nodos del supergrid, `z_to_dz`), TPXO10 con el de
  `write_tpxo_boundary.py` (velocidad propia de TPXO, u = U/h_TPXO, R19; elipse rotada con `angle_dx` = 0 aquí).
- ERA5 para `data_override`: meses concatenados, latitud sur → norte, sin empaquetar; humedad específica de `d2m` y
  **`msl`** (CEFI usa `sp`, que no se bajó), lluvia = `tp` y nieve 0 (no se bajó `sf`; verano) — clase c.
- ríos (R8): el `rivers.nc` del caso (regla ldd de CEFI) como runoff Q·1000/A_t; MOM6 lo vuelve volumen con
  ρ₀ = 1035, pyOCEAN suma Q: el volumen de ríos de MOM6 es 1000/1035 del de pyOCEAN (clase c).

**Diferencias con NWA12 (`MOM_override`).** Malla 87 × 89 y cuatro segmentos; `OBC_HOR_INDEXING_BUG = False` (R6);
sin COBALT; estado inicial común (R24, abajo); `TIDE_REF_DATE` 2017-05-01, `TIDE_NODAL_REF_DATE` 2017-07-01 (R3).

**Onda corta (R23, R25).** NWA12 usa MANIZZA_05 (3 bandas, opacidad según la clorofila). En la configuración de
CEFI 2025_04 con COBALT, `CHL_FROM_FILE = False` porque la clorofila la da COBALT; la configuración **sólo física**
de NWA12 (settings de Zenodo 7893727, `MOM_input:639-644`) usa `VAR_PEN_SW = True`, `CHL_FROM_FILE = True`,
`CHL_FILE = "seawifs-clim-1997-2010.nwa12.nc"`, `CHL_VARNAME = "chlor_a"`, 3 bandas: eso es lo que corre la referencia
`mom6_full`. El archivo de NWA12 no es público (sólo en ncrc de NOAA), así que `mom6_inputs.write_chl` lo hace como
CEFI `tools/opacity/regrid_opacity.py` desde la misma fuente (`seawifs-clim-1997-2010.1440x1080.v20180328.nc`, la
climatología SeaWiFS de OM4, `~/pyOCEAN_refs/nwa12_grid/`): inundación de tierra (`flood_kara`, :26; aquí el sustituto
de 8 vecinos) y regrid **conservativo** de las esquinas de las celdas fuente a las del supergrid (:10-15, 28-36),
`chlor_a(time, yh, xh)` con `area`, `lon_crnr`, `lat_crnr`, ejes enteros y el eje de tiempo mensual `modulo` (:37-55).
Sin xesmf: alrededor del recorte las dos mallas son rectilíneas exactas en lon/lat (lo comprueba el código), así que
cada solapamiento es un rectángulo esférico R² Δλ Δ(sen φ) y los pesos conservativos son exactos y separables; ESMF
usa lados de círculo máximo (diferencia de segundo orden, clase c). Clorofila de mayo–agosto en el recorte: media
1.80 mg m⁻³, rango 0.07–13.4. El **control de paridad** `mom6_full_jerlov` usa la onda corta de pyOCEAN: doble
exponencial de Jerlov I (0.58 a 0.35 m, 0.42 a 23 m). pyOCEAN no tiene la onda corta de Manizza ni clorofila: **clase
(b)**, para el proyecto 4 (con 0.5–3 mg m⁻³ en la plataforma, Jerlov I deposita el calor demasiado hondo). `pgf0` y
`tides` usan Jerlov (no entra en el PGF del primer paso y el modo de mareas no tiene onda corta).

**Estado inicial común (R24).** Los dos modelos arrancan del mismo archivo, `ic_full.nc` = el estado inicial de
pyOCEAN (h, θ, S_P, u, v); `write_glorys_initial.py` de CEFI no se usa (con él MOM6 remapearía GLORYS por su cuenta y
el criterio 7, mismo estado inicial, no se podría cumplir). Una corrida de MOM6 con el estado inicial de CEFI sería
sólo una sensibilidad opcional.

**Ríos (R8, R22).** Los dos modelos usan el mismo `rivers.nc` del caso (regla ldd de CEFI), no
`tools/rivers/write_runoff_glofas.py` (necesita xesmf y GloFAS de todo NWA12; sólo se bajó la caja del recorte); MOM6
convierte el flujo de masa Q·1000/A_t en volumen con ρ₀ = 1035, así que su volumen de ríos es 1000/1035 del de
pyOCEAN (R22, clase c).

**FMS 16bb3af (= 2026.01.01).** Decisión del controlador para la Tarea 10: el commit del submódulo de CEFI, en
`~/pyOCEAN_refs/FMS_2025.03` (el nombre del directorio viene del brief; `git describe` dice 2026.01.01), porque FMS
2025.02.01 no exporta `fms_error_mesg`/`fms_check_nml_error` que usa el acoplador 175a9da.

**Configuración efectiva contra NWA12.** El `MOM_parameter_doc.all` de la corrida del recorte comparado con el de
NWA12 (revisión de la Tarea 10) sólo difiere en lo de arriba: NIGLOBAL/NJGLOBAL, el segmento 4 y su `_DATA`,
`OBC_HOR_INDEXING_BUG`, `USE_generic_tracer` (y los parámetros de COBALT que dependen de él), las fechas de marea (y
las fases `TIDE_*_PHASE_T0` derivadas), el estado inicial desde archivo (`THICKNESS_CONFIG`, `TS_FILE`, `SALT_FILE`,
`VELOCITY_FILE`, `DEPRESS_INITIAL_SURFACE`, `REMAP_AFTER_INITIALIZATION`), la onda corta del modo y
`DATE_STAMPED_STDOUT`; más `PHA_MLD_*`, parámetros del fork de CEFI que el MOM6 de GFDL 0e9a561 no tiene (sólo un
diagnóstico). Así que MOM6 0e9a561 corre el mismo juego de parámetros que NWA12. `BETTER_BOUND_AH`,
`CFL_BASED_TRUNCATIONS`, `USE_DIABATIC_TIME_BUG`, `HREF_FOR_MLD` y los `OBC_*_STRAIN/VORTICITY` viejos, que MOM6 avisa
como no usados u obsoletos, **tampoco están en el `MOM_parameter_doc.all` de NWA12**: son entradas viejas del
`MOM_input` de NWA12 que NWA12 mismo no usa, no una diferencia de versión.

Modo `tides`: océano homogéneo (10 °C, 35) en reposo, segmentos sólo con marea (`value:` para U, V, SSH, T, S),
`WIND_STRESS_MULTIPLIER = 0` y una atmósfera constante en equilibrio con 10 °C (`data_table_tides`: sin viento ni
onda corta ni lluvia, aire a 10 °C, q = 0.98 q_sat, onda larga σT⁴); T se queda en 10.0000 los 30 días
(`ocean.stats`). Modo `pgf0`: un paso acoplado de 600 s con `TIDES` y `CALCULATE_SAL` apagados y
`SAVE_INITIAL_CONDS` (`MOM_IC.nc`: `check_mom6.ic_check` compara h, T, S, u y v que MOM6 leyó con `ic_full.nc`).

### Resultados MOM6 (2026-11-02)

- **Criterio 1 ✅:** las 23 variables estáticas de `ocean_static.nc` (geolon/lat de T, u, v, q; dxt, dyt, dxCu, dyCu,
  dxCv, dyCv, dyCuo, dxCvo, area_t/u/v/q, Coriolis, cos_rot, sin_rot) son **iguales bit a bit** a
  `grid_from_supergrid` en el recorte (error relativo máximo 0). `f_t` no entra (R9); dxBu, dyBu no son
  diagnósticos estáticos de MOM6.
- **Criterio 7 (PGF inicial) ✅:** con el mismo archivo inicial, max |PFu − pyOCEAN| = 1.16e-14 y |PFv − pyOCEAN| =
  1.22e-14 m s⁻² (|PF| ≤ 1.9e-4) en todas las caras interiores mojadas y todas las capas. pyOCEAN deja fuera −g∇η
  (es del solver barotrópico) y se suma para comparar. **Hay que usar los ajustes de NWA12:** `g` = 9.8 (G_EARTH) y
  `pgf_mass_weight='full'` (MASS_WEIGHT_IN_PRESSURE_GRADIENT = True, VANISHED_ONLY = False). Con lo que
  `parity_config()` usa hoy (g = 9.81 por defecto, 'vanished') la diferencia es 5.2e-7 m s⁻²: dos diferencias de
  paridad de la Tarea 9 (clase a de la configuración del caso, no del núcleo).
- **Mareas solas (30 días, 600 s, sin rampa):** error vectorial de M2 en 11 mareógrafos (`tide_check.py`, días
  1–30), **MOM6 0.218 m de media (mediana 0.193)**, pyOCEAN 0.229 m (0.200). **Superado (Tarea 11):** la SSH horaria
  de MOM6 es el promedio de los 6 valores de fin de paso de la hora de acople, no una instantánea (MOM6 0e9a561
  `MOM.F90:1050-1100`); leída como tal, MOM6 llega 25 min tarde (12° en M2). `tide_check.py` deshace ese promedio
  desde `af63946`: MOM6 0.244 m de media (0.240 mediana), pyOCEAN con la biarmónica 0.224 (0.193). MOM6 tiene mejor la fase en la bahía de
  Nueva York (Sandy Hook +1.8° contra −18.6°, Battery −8.6° contra −31.3°) y peor la amplitud en Long Island Sound
  (Bridgeport 0.57 contra 0.92 m, obs. 0.98). La clasificación (criterio 4) es de la Tarea 11.
- **Corrida completa de MOM6 con Jerlov I (`mom6_full_jerlov`, control de paridad, R23) ✅ (criterio 7, primera parte):** 123 días desde el 2017-05-01, mismas entradas, DT 600
  / DT_THERM 1800 de NWA12 sin cambios: termina sin truncamientos (`ocean.stats`, CFL máx 0.28), salidas diarias 3-D y
  η horaria. La mirada a SST contra OISST y a η en mareógrafos está en la Tarea 11 (abajo).
- **Corrida de referencia `mom6_full` (MANIZZA_05, R23, R25):** 123 días, DT 600 / DT_THERM 1800, 0 truncamientos, CFL máx 0.26, 26.0 s por día simulado en 5 rangos.
  SST media del dominio (tos diaria, ponderada por área): MANIZZA − Jerlov = **+0.30 °C el día 123** (24.03 contra
  23.73 °C); −0.10 °C el día 31, +0.06 el día 93, +0.05 de media en los 123 días. Es el sentido esperado: con
  clorofila de plataforma el calor queda más arriba al final del verano (Jerlov I lo deposita más hondo).
- **MOM6 lee el estado inicial común bit a bit (revisión M3) ✅:** `MOM_IC.nc` de `pgf0` (`SAVE_INITIAL_CONDS`) contra
  `ic_full.nc`: |Δ| = 0 en h, T, S (columnas mojadas) y u, v (caras interiores mojadas).
- **Costo en 5 núcleos P (5 rangos MPI):** MOM6 + SIS2 **26.4 s por día simulado** (`mom6_full_jerlov`) (123 días en 54 min); mareas solas
  27.4 s/día; pyOCEAN, la misma corrida completa a Δt 300 s con 5 hilos, 208.5 s/día (7.9× MOM6; con Δt 600 sería
  ≈ 106 s/día, 4×).

**Qué debe mejorar pyOCEAN (de esta tarea):** (1) ~~`parity_config()` debe pasar `g=9.8` y
`pgf_mass_weight='full'`~~ hecho (R21); (2) el costo por día (×8 contra MOM6 en los mismos núcleos): ~~Δt 300 por la
inestabilidad de θ_bc = ½~~ (corregida, `e1ffbbb`: ×4 a Δt 600) y la parte en serie del paso (1.6× de 1 a 5 hilos);
(3) la fase de M2 en la bahía de Nueva York: con la SSH de MOM6 bien leída, MOM6 también se adelanta (Battery −20.7°,
Sandy Hook −10.4°); pyOCEAN 8–10° más (Tarea 11).

Reproducir: `python validation/nwa12_cut/mom6_inputs.py` → `validation/nwa12_cut/build_mom6.sh` →
`validation/nwa12_cut/run_mom6.sh pgf0|tides|full|full_jerlov` → `python validation/nwa12_cut/check_mom6.py --static --pgf
--series --cost RUNS/runs/mom6_tides RUNS/runs/mom6_full RUNS/runs/mom6_full_jerlov` → `python validation/nwa12_cut/tide_check.py
RUNS/runs/mom6_tides`.

## Tarea 11: comparación con observaciones, cierre de la cadena (2026-11-03)

Detalle y tablas completas en `RUNS/figs/obs_qc/README.md` (control de calidad), `RUNS/figs/obs_compare/README.md`
(comparación, con OISST en su §5) y `RUNS/figs/gs_diag_*/README.md` (Corriente del Golfo). Aquí, lo que cierra la
cadena.

### Correcciones del núcleo que salieron de la cadena (pausa de la Tarea 11, 2026-11-02/03)

La corrida de paridad de 123 días a Δt 300 no se podía puntuar: desde el día ~40 un modo cuasi-inercial llevaba la
capa superior de la Corriente del Golfo a 5.5–7.2 m/s (`figs/gs_diag/README.md`). Orden aprobado por Esteban y
resultado:

1. **Flujo de PV / θ_bc = ½ (`e1ffbbb`, clase a).** El flujo de PV del momento vector-invariante usaba el transporte
   del paso anterior; MOM6 sólo lo hace en su predictor (`MOM_dynamics_split_RK2.F90:566, 793, 1103`). Ahora usa uⁿ,
   hⁿ. La caja inercial decae (−0.16 /d en vez de crecer), DOME a 1200 s queda estable en z\*, σ e híbrida, la onda
   interna de Ilicak queda igual (0.736 / 0.579 / 0.261), y la paridad corre a Δt 600 (`docs/design/theta_bc_inertial/root_cause.md`).
2. **Velocidad tangencial de MOM6 en los bordes abiertos (`b48d367`, paridad `2f80d5a`, R27):** fila propia en
   `CLAUDE.md`.
3. **Viscosidad horizontal en las caras abiertas (`ca36b59`, clase a).** Con `obc_tangential='none'` la cara abierta
   era una pared en la tensión de Smagorinsky y en la laplaciana de A_h ((u₁ − 0)/Δx). Ahora entra su velocidad
   normal, como MOM6 (`mask2dCu` = 1 en las caras OBC, `MOM_grid_initialize.F90:1270-1288`; sin rama OBC en la
   tensión, `MOM_hor_visc.F90:708-715`). La paridad (`'mom6'`) queda bit a bit; DOME la siente (abajo).
4. **Smagorinsky biarmónica de MOM6 (`a1a2c14`, paridad `1789e9f`).** El ancho del chorro en 72 °W pasa de 96 km
   (laplaciana 0.15) a **72 km** (mediana de los días 60–123; MOM6 92, GLORYS 73), sin costo extra (0.99×), 0
   truncamientos. La cota `BOUND_AH` × 6/11 **no** causa el tablero de ajedrez de ζ (`figs/gs_diag_bihbound`: nunca
   actúa en la capa de superficie ni en la región de la métrica). Comparado media diaria con media diaria
   (`figs/gs_diag_dailymean`, corrida horaria `full_dt600_bih30h`), el tablero es ζ/f **0.0286** contra 0.023 de MOM6
   (1.24×, c, pequeño) y SST **0.089** contra 0.088 °C (igual): las instantáneas de las 00 UTC explicaban todo el
   exceso de SST y ~55 % del de ζ.

Corrida vigente `full_dt600_bih123` (1789e9f; Δt 600, 123 días, 5 hilos): 0 truncamientos, 0 eventos del limitador,
max|u| 2.98 m/s, 105.2 s por día simulado; presupuestos relativos al contenido / al flujo bruto: volumen 1.5e-16 /
6.8e-16, calor 1.2e-14 / 2.8e-14, sal 5.3e-14 / 2.4e-13 (`rel_terms` 1.0e-12 / 5.2e-13 / 7.8e-11, la sal por
redondeo como en la Tarea 9).

### Control de calidad de las observaciones (paso 0; `obs_qc.py`, `e96419e`; regla de Esteban R26)

| registro | veredicto | razón |
|---|---|---|
| 13 mareógrafos CO-OPS (6 min y horario, verificados) | 10 **usables**, 3 **parciales** (Newport, Bridgeport, Montauk: bloques inferidos por CO-OPS y el hueco de 264 h de Montauk), 0 corruptos | sin escalones de datum ni fallas de reloj confirmados; los 23 candidatos a escalón desaparecen en la diferencia con los vecinos (señal oceánica); el registro de 2017 = sus constantes `harcon` a 0.2–2.2 cm en M2 |
| TPXO10 en los mareógrafos | no representativo en Woods Hole, Nantucket, Montauk y The Battery (d) | se reportan aparte |
| NDBC 44025 (WTMP, ATMP, PRES, WSPD) | **parcial** (86.6 % de WTMP) | huecos de 26 h (30 may) y 293 h (29 jul – 10 ago); 8 picos, 47 h pegadas; sin salinidad ni T subsuperficial en 2017 |
| OISST v2.1 AVHRR (124 días, `compare.oisst_qc`) | **usable, parcial**: 18 de 713 píxeles de agua fuera | completo, versión final, sin hielo ni valores fuera de rango, `err` mediana 0.21 °C; 2 píxeles de Long Island Sound +3.4/+5.0 °C sobre GLORYS (anomalía > 5 °C 35–64 días) y 16 costeros sin referencia; +0.53 °C sobre la boya 44025 por representatividad (el sesgo crece hacia la costa en los cuatro píxeles del estencil, d) |

### Resultados (pyOCEAN = paridad vigente; MOM6 = `mom6_full` MANIZZA_05 y el control `mom6_full_jerlov`)

| comparación | pyOCEAN | MOM6 | MOM6 Jerlov | otra referencia |
|---|---|---|---|---|
| **mareas solas, M2 vectorial mediana / media [m], 11 mareógrafos** (criterio 4; `tides_dt600_bih`) | **0.193 / 0.225** | 0.241 / 0.245 | — | TPXO10 0.062 / 0.148 |
| idem, 7 mareógrafos donde TPXO10 es representativo | 0.184 / 0.166 | 0.181 / 0.202 | — | TPXO10 0.045 / 0.041 |
| η, 4 meses: submareal menos barómetro inverso, RMSE [m] / r (mediana de 11) | 0.047 / 0.85 | 0.052 / 0.76 | 0.049 / 0.76 | — |
| SST contra OISST, plataforma interior Jun–Ago, sesgo / RMSE [°C] | +1.07 / 1.80 (≈ +0.85 sin el sesgo de muestreo) | +0.51 / 1.53 | +0.11 / 1.44 | GLORYS12 +0.10 / 1.07 |
| SST contra OISST, agua profunda interior Jun–Ago | +2.46 / 3.37 | +2.07 / 3.01 | +2.31 / 3.19 | GLORYS12 +0.42 / 1.50 |
| SST contra GLORYS12, plataforma interior Jun–Ago | +0.95 / 1.86 | +0.36 / 1.75 | −0.00 / 1.68 | — |
| SST en 44025, medias diarias, sesgo / RMSE / r | +0.49 / 0.77 / 0.994 | +0.82 / 1.14 / 0.987 | +0.19 / 0.78 / 0.989 | GLORYS +0.65; OISST +0.53 / 0.74 / 0.994 |

Comparación de la SSH de MOM6: su salida horaria es el promedio de la hora de acople (arriba, Tarea 10); `compare.py`
y `tide_check.py` lo deshacen. Sesgo de muestreo de pyOCEAN: la corrida diaria guarda instantáneas de las 00 UTC
contra las medias diarias de MOM6, GLORYS y OISST; medido contra OISST con la corrida horaria, +0.11 °C en la
plataforma en mayo y ≈ +0.2 °C en verano (por la serie por paso en los 13 puntos), +0.02 °C en agua profunda.

### Observaciones in situ: QC y comparación (OOI Pioneer, planeadores, eMOLT; `obs_qc_insitu.py`, `compare.py --only profiles moorings emolt adcp intrusion`)

QC primero (R26; `diagnostics/profile_qc.py`, `RUNS/figs/obs_qc/README.md` §3): **0 registros corruptos**. 19
despliegues de planeador: 16 usables, 3 Spray parciales (posiciones marcadas por el proveedor); 14 de los 19 no traen
banderas útiles (los 9 OOI diferidos no tienen variables de QC y las de los tiempo real están en NaN), así que
deciden nuestras pruebas: la sal profunda S(θ = 6 °C) de 10 despliegues es 35.046–35.064 con deriva ≤ 0.014, y los
pares entre plataformas (5 km, 2 h) no muestran sesgos > 0.2 °C / 0.05. 5 perfiladores y 7 ADCP usables, 6 CTD
fondeados (4 usables, 2 parciales por huecos; dos redespliegues a otra profundidad), velpt y presf parciales y no
usados (la presión de velpt está en kPa rotulada dbar). eMOLT: 12 sitios usables, 6 representables en el recorte (5
al norte de 42 °N, 2 en bahías donde el modelo tiene 5–6 m). Comparación con 29 350 perfiles (junio–agosto):
**la hipótesis de la capa de mezcla (KPP frente a ePBL) queda refutada**: en la plataforma la mediana de la MLD es 7.6 m
en pyOCEAN, 7.6 en las observaciones y 6.2 en MOM6 Jerlov (pyOCEAN más somera sólo en 26 % de los perfiles); el
exceso de calor llena los 30 m superiores (θ medio 0–30 m 21.8 contra 17.0 °C observado y 18.8 de MOM6 Jerlov) y
viene con sal (S 34.9 contra 32.4; correlación Δθ–ΔS frente a MOM6 Jerlov r = 0.88): es advección lateral. pyOCEAN
inunda desde el día ~45 el 69 % del fondo de la plataforma interior del sur del MAB (H < 60 m) con agua de ≈ 21 °C y
S > 35 (MOM6 28–43 %, 11–16 °C; GLORYS ≤ 6 %; observado: piscina fría de 7–11 °C), al mismo tiempo que la deriva al
norte de la pared de la Corriente del Golfo: en la plataforma interior de Delaware el sesgo es +7.7 °C / +3.5 a
10–30 m (MOM6 −1.2 / +1.6). En el resto los dos modelos se parecen: termoclina de la plataforma externa y del mar de
talud +4…+6 °C a 30–60 m en los tres (GLORYS +1.8), NSIF de 7 m +1.5…+2.3 °C, fondo de eMOLT +1.8 / +1.5 / +0.7 °C
(mediana pyOCEAN / MOM6 / MOM6 Jerlov), y los dos invierten la corriente media sobre la plataforma en OOI (ADCP
−0.07 m/s hacia el este, modelos +0.02…+0.05, GLORYS −0.04). Clases: intrusión (b/c, trayectoria libre de la
Corriente del Golfo, magnitud propia de pyOCEAN), corriente invertida (c de la configuración del recorte,
[hipótesis: entrada a lo largo de la plataforma por el borde noreste]). Detalle en `RUNS/figs/obs_compare/README.md` §6.

### Criterios del spec

| criterio | estado | dónde |
|---|---|---|
| 1. malla = MOM6 compilado | ✅ 23 variables estáticas bit a bit | Tarea 10 |
| 2. invariancia en mallas curvilíneas | ✅ ≤ 1e-10 en los 4 casos; Coriolis de flujo σ corregido (a) | Tarea 3, `docs/design/regional_refs/metric_audit.md` |
| 3. GLORYS | ✅ lectura = archivo crudo, rotación exacta, contenido por columna, caras del borde | Tarea 7 |
| 4. mareas solas | ✅ M2 0.193 m contra 0.241 de MOM6 (empate en los 7 mareógrafos buenos de TPXO10); los dos 3× TPXO10 | arriba |
| 5. presión atmosférica | ✅ barómetro inverso estático a 3.4e-5 (σ) y 1.1e-5 (z\* esférica), volumen exacto, apagada = bit a bit | Tarea 6 |
| 6. corrida completa de paridad | 🔶 sin truncamientos ni eventos del limitador, presupuestos a redondeo, restart y 1 = 5 hilos bit a bit (corrida de la Tarea 9 y pruebas del caso pequeño con la paridad vigente); **costo 105 s/día contra 26.0 de MOM6 (4.0×)** | arriba |
| 7. MOM6 en el recorte | ✅ 123 días con las mismas entradas, PGF inicial a 1.2e-14, primera mirada a SST (GLORYS, OISST, 44025) y a η | Tarea 10, arriba |
| 8. sin cambios en lo existente | ✅ con regeneraciones deliberadas: las correcciones (a) `e1ffbbb` (flujo de PV: 17 casos ALE vector-invariantes) y `ca36b59` (caras abiertas: 5 casos con fronteras abiertas y ah > 0) regeneraron a propósito los goldens que tocan; la de Coriolis σ (`5d8aab4`) dejó los goldens bit a bit, cada una con su máx. \|Δ\| en el commit | `CLAUDE.md` |

### Clasificación de lo que queda

- **Fase de M2 en la bahía de Nueva York y Long Island Sound** (los dos modelos 10–38° adelantados; pyOCEAN 8–10°
  más que MOM6): batimetría de 1/12° de los estrechos y puertos y sin `CHANNEL_DRAG` (b/c). Fuera de costa pyOCEAN va
  4–5° adelante de TPXO10 y MOM6 1–3° atrás [hipótesis: paso barotrópico implícito θ frente al split-explicit de
  MOM6 con η filtrada; c hasta probar lo contrario].
- **Calentamiento de la plataforma en verano** (≈ +0.75 °C sobre MOM6 Jerlov, con la misma onda corta, bulk y estado
  inicial; confirmado contra OISST y GLORYS): **no** es KPP frente a ePBL (refutado con los perfiles, arriba); es agua
  de talud / de la Corriente del Golfo sobre la plataforma desde el día ~45 (b/c); MANIZZA_05 vale ≈ 0.4 °C en MOM6
  (b para pyOCEAN, que no la tiene).
- **Agua profunda +2 °C en los tres modelos**: trayectoria de la Corriente del Golfo y anillos en una corrida libre
  (c); la pared norte en 72 °W deriva al norte desde el día ~50 en pyOCEAN (38.9 °N, MOM6 37.8, GLORYS 37.4), sin
  atribuir con una sola realización (b/c).
- **η submareal**: sin presión atmosférica en la paridad (`MAX_P_SURF` = 0 en NWA12); con el barómetro inverso
  quitado de la observación, los dos modelos tienen 11–25 % de varianza de más (b contra la verdad, c de la paridad).
- **Costo 4× MOM6**: la parte en serie del paso (1.6× de 1 a 5 hilos) y un solo Δt (MOM6 separa `DT_THERM` 1800).

### Qué debe mejorar pyOCEAN (cadena regional)

1. La intrusión de agua de la Corriente del Golfo en la plataforma interior (69 % del fondo al sur de 40.5 °N con
   S > 35 contra 28–43 % en MOM6): presupuesto del flujo de calor y sal a través de la isóbata de 60 m frente a
   MOM6, y la entrada a lo largo de la plataforma por el borde noreste (los dos modelos invierten la corriente en
   OOI). La capa de mezcla de KPP no es la causa (MLD = observada); ePBL queda como opción del proyecto 4.
2. Onda corta de Manizza con clorofila (b): vale ≈ 0.4 °C de SST de plataforma en MOM6.
3. Arrastre de canal / subgrid (`CHANNEL_DRAG`) y constituyentes de aguas someras (M4, MS4, M6) para Long Island Sound.
4. Presión atmosférica encendida en el proyecto 4 (`atm_pressure=True` existe; la paridad la deja apagada).
5. Costo: menos parte en serie por paso y un Δt de trazadores separado, para acercarse a los 26 s/día de MOM6.
6. Salida de medias diarias de SST (y SSH horaria) para comparar media con media; hoy se corrige con la corrida horaria.
7. La deriva de la Corriente del Golfo: estadística de varias realizaciones en el proyecto 4, no una sola trayectoria.

Reproducir la comparación: `python validation/nwa12_cut/obs_qc.py` → `python validation/nwa12_cut/compare.py`
(`--only tides eta sst buoy oisst`); salidas en `RUNS/figs/obs_qc/` y `RUNS/figs/obs_compare/`.
