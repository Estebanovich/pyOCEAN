# Recorte de NWA12: caso regional real (cadena regional, Tarea 9)

- **Spec:** `docs/superpowers/specs/2026-11-01-regional-chain-design.md` (criterios 4, 6 y 8; "Decisiones del caso")
- **Plan:** `docs/superpowers/plans/2026-11-01-regional-chain.md`, Tarea 9
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
  cuenca cerrada).

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
| momento | vector-invariante; Smagorinsky biarmónica `SMAG_BI_CONST` 0.03 | vector-invariante; Smagorinsky **laplaciana** 0.15 (OM4) | (b) sin biarmónica |
| EOS | `WRIGHT_FULL` (θ, S_P), ρ₀ 1035, `C_P` 3992 | `eos='wright_full'`, `ts_kind='theta_sp'`, ρ₀ 1035, c_p 3992 | — |
| superficie | bulk NCAR del acoplador FMS, con SIS2 | `BulkFluxes('ncar')` (= `surface_flux` de FMS bit a bit), ERA5 con `lsm` | (c) sin SIS2, sin ráfagas |
| onda corta | `VAR_PEN_SW` con clorofila (COBALT), 3 bandas | Jerlov I, Paulson & Simpson | (c) |
| mezcla vertical | ePBL + Jackson + doble difusión, `KD` 1e-6, `KV` 0 | KPP (CVMix) + Jackson, κ_bg 1e-6 (el `KD` de NWA12, pasado como parámetro; el valor por defecto de pyOCEAN sigue en 0), ν_bg 0 | (b) sin ePBL ni doble difusión |
| fricción de fondo | cuadrática `CDRAG` 0.003, `HBBL` 10 m, `DRAG_BG_VEL` 0, `CHANNEL_DRAG` | cuadrática cd 0.003, hbbl 10, U_bg 0 | (b/c) sin `CHANNEL_DRAG` (R5) |
| bordes abiertos | Flather + Orlanski + nudging 3 d / 360 d (según la velocidad de fase de Orlanski), reservorio de trazadores de 9 km | `'radiating'`: Flather + Orlanski + nudging τ_in 3 d / τ_out 360 d (según el sentido del flujo, R7), sin reservorio ni esponja | (c) |
| exterior | GLORYS diario en sus niveles (`z_to_dz` de CEFI), remapeado con PPM_H4 | GLORYS diario en sus niveles (`source_level_grid`, interfaces a medio camino), remapeado a las capas, lineal en el tiempo; ū = media de la columna de la celda de borde | — |
| marea de borde | TPXO, 10 constituyentes, η y u sumados al segmento, nodal fijo en una fecha | `BoundaryTides` con TPXO10 (10 constituyentes), u = U/D del modelo (R3), nodal OTPS en cada instante, sin rampa | (c) |
| marea de potencial | `TIDES`, SAL escalar 0.05, SAL en Flather | `tidal_potential=True`, `sal_beta` 0.05, `sal_flather` | (c) camino SAL antiguo de MOM6 (R4) |
| presión atmosférica | apagada (`MAX_P_SURF` = 0) | `atm_pressure=False` | — |
| ríos | GloFAS v3 con la regla de CEFI; mezcla de ríos de ePBL (`DO_RIVERMIX`) | GloFAS v4 con la regla `ldd` de CEFI (R11, R12), agua dulce real en la capa superior | (c) versión; (b) sin mezcla de ríos |
| Δt | `DT` 600 s, `DT_THERM` 1800 s | un solo Δt, el estable más grande medido (600, luego 300) | (c) |
| estado inicial | T, S, SSH y velocidades de GLORYS, remapeo desde z | T, S remapeados desde los niveles de GLORYS (contenido exacto por columna), η = zos (sin quitar la media, como MOM6), u, v rotados e interpolados linealmente en la vertical a cada cara | (c) velocidades interpoladas, no remapeadas |

**Por qué ALE z\* + momento vector-invariante + Smagorinsky.** El momento en forma de flujo de σ y la laplaciana de
A_h constante no tienen los términos métricos curvilíneos (Tarea 3, `docs/design/regional_refs/metric_audit.md`
§2.3, clase b). Las formas invariantes sí son correctas en una malla curvilínea, como en MOM6. `parity_config()`
rechaza otra combinación, y el pre-flight avisa (`PyOceanConfigWarning`) si una malla de supergrid se usa con la
forma de flujo o con la laplaciana constante.

**Decisiones aceptadas (R15, R16).** Δt = 300 s para la corrida completa: es la inestabilidad conocida de
θ_bc = ½ (clase a, con su plan aparte; MOM6 corre con 600 s), no un ajuste de física. κ_bg = 1e-6 es el valor de
paridad con NWA12; el punto (2) de "qué debe mejorar" de N2b prefiere decidir la mezcla de fondo contra la boya
44025, así que se revisa en el proyecto 4. Sin banda de esponja (NWA12 no tiene) y velocidades iniciales
interpoladas en la vertical (MOM6 las remapea): aceptadas, clase c.

**Física de NWA12 que pyOCEAN no tiene (clase b o c, para el proyecto 4):** ePBL, doble difusión, Smagorinsky
biarmónica, `CHANNEL_DRAG`, el reservorio de trazadores de los segmentos (9 km), la mezcla de ríos de ePBL y SIS2.

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
python validation/nwa12_cut/run_pyocean.py --mode full --days 123 --dt 300 --threads 5 --restart-at 60
```

Δt = 300 s es el estable medido (abajo); tarda ≈ 6.8 h. Para el chequeo del restart a la mitad:
`--start-from …/runs/full_dt300/restart_day60.nc --tag from_restart` y luego `--compare` de los dos `final.nc`.

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
local de KPP) conserva el contenido. El residuo acumulado se mide contra el flujo bruto Σ|términos| (la medida
estricta) y contra el contenido.

## Resultados (2026-11-01; Tarea 9)

Todas en 5 núcleos P (`NUMBA_NUM_THREADS=5`), 87 × 89 × 75.

| corrida | Δt | días | truncamientos CFL | eventos del limitador | puntos limitados de Smagorinsky | costo |
|---|---|---|---|---|---|---|
| mareas solas | 600 s | 30 | 0 | 0 | 26 222 (≈ 6 por paso) | 0.675 s/paso, 97 s por día simulado |
| completa (paridad) | 600 s | 30 (mayo; ERA5 cubría solo mayo) | 0 | 0 | 25 218 | 0.736 s/paso, 106 s/día |
| completa (paridad) | 300 s | 30 | 0 | 0 | 0 | 0.689 s/paso, 198 s/día |

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

**Presupuestos del dominio** (residuo acumulado / flujo bruto; criterio ≤ 1e-10):

| corrida | volumen | calor | sal |
|---|---|---|---|
| completa, 600 s, 30 días | 8.0e-15 | 8.5e-14 | 3.0e-13 |
| desde el restart del día 15, 600 s | 8.7e-15 | 7.3e-14 | 2.9e-13 |
| completa, 300 s, 30 días | 1.2e-14 | 2.2e-13 | 1.0e-12 |

Relativos al contenido: ≤ 6e-14. El peor paso suelto llega a 3e-7 del bruto de ese paso, en pasos con flujo neto
casi nulo (marea en repunte): es el redondeo de las sumas del contenido (ε · 5e14 m³ ≈ 0.06 m³), no un término que
falte.

**Restart y hilos:** 30 días seguidos contra 15 días + restart + 15 días (600 s, con GLORYS, mareas, ERA5 y ríos
variando): las 19 variables del restart final y todas las series por paso de las estaciones son idénticas bit a bit.
1 contra 5 hilos en 3 horas de la corrida completa: idénticos (η, u, v, T, S, h, estado del cierre, contadores).

**Mareas (vistazo, sin puntaje; el criterio 4 se juzga contra MOM6 en la Tarea 10):** M2 en 11 mareógrafos de
CO-OPS, días 1–30 de la corrida de mareas solas: error vectorial medio 0.23 m (mediana 0.20 m); TPXO10 en los mismos
puntos, 0.15 m (casi todo Nantucket y Woods Hole, que la malla de 1/12° no resuelve). Las amplitudes están cerca
(Bridgeport 0.93 contra 0.98 m, Atlantic City 0.63 contra 0.58), pero la fase de pyOCEAN se adelanta 12–31° en la
bahía de Nueva York y Long Island Sound (Sandy Hook −19°, Battery −31°, Montauk −27°, Bridgeport −12°) y solo
2–4° en Atlantic City y Lewes. No se afina nada; se compara con MOM6 en el mismo recorte.
