# Cadena regional: recorte de NWA12 con fronteras abiertas reales, mareas, presión atmosférica, ríos y MOM6 en el mismo recorte

**Fecha:** 2026-11-01. **Estado:** escrito sin sesión de preguntas, por pedido de Esteban ("Sin preguntas", 2026-11-01). Toda decisión que no venga de un acuerdo anterior está marcada **[decisión de Claude]** y se puede revertir.

**Contexto:** `main` en `91d1ab2` (2134 tests). Es el tercero de los cuatro proyectos que cierran N2 (`docs/design/vision_roadmap.md` §8, acuerdo del 2026-10-31): N2b ✅ → TEOS-10 en 3D ✅ → **cadena regional (este spec)** → benchmark NWA12. El alcance acordado el 2026-10-31 para este proyecto es:
- fronteras abiertas de T, S, u, v y η que varían en el tiempo, desde GLORYS;
- mareas 3D en la frontera;
- malla y batimetría del recorte de NWA12;
- presión atmosférica;
- MOM6 corrido en el mismo recorte.

Lectura de referencia: NWA12 de CEFI en `~/pyOCEAN_refs/CEFI-regional-MOM6` (git `36b8f49`), sobre todo `exps/NWA12.COBALT/MOM_parameter_doc.short` y `tools/{boundary,initial,atmos,rivers,grid}`. La lectura del código fuente de MOM6 que haga falta (`MOM_grid_initialize`, `MOM_open_boundary`, `MOM_tidal_forcing`, `MOM_surface_forcing`) va a `docs/design/regional_refs/`, con `archivo:línea`, antes de implementar cada parte. Hechos de NWA12 que este spec usa:
- **Malla:** 775 × 845, de un supergrid (`ocean_hgrid.nc`) y `ocean_topog.nc`.
- **Vertical:** z\* con 75 niveles de `vgrid_75_2m.nc` (está en local, `tools/grid/`).
- **Pasos de tiempo:** `DT` 600 s, `DT_THERM` 1800 s.
- **Fronteras abiertas:** tres segmentos (sur, norte, este), cada uno `FLATHER, ORLANSKI, NUDGED, ORLANSKI_TAN, NUDGED_TAN`.
  - Nudging de velocidad: 3 días hacia adentro y 360 días hacia afuera.
  - Reservorio de trazadores de 9 km (`OBC_TRACER_RESERVOIR_LENGTH_SCALE_IN/OUT`).
  - Datos de GLORYS diarios (`thetao`, `so`, `zos`, `uv`) en el supergrid (`BRUSHCUTTER_MODE`).
- **Mareas:** 10 constituyentes en la frontera, M2, S2, N2, K2, K1, O1, P1, Q1, MM y MF (TPXO, amplitud y fase de η y de u, v), con fase de equilibrio y corrección nodal; también marea de potencial (`TIDES = True`).
- **Presión atmosférica:** **apagada.** `MAX_P_SURF = 0.0` recorta la presión de superficie a cero.
- **Atmósfera:** ERA5 (`tools/atmos`).
- **Ríos:** GloFAS (`glofas_runoff_*.nc`, `tools/rivers`).
- **Física:**
  - Mezcla vertical: ePBL, Jackson y doble difusión.
  - Viscosidad horizontal: Smagorinsky biarmónica (`SMAGORINSKY_AH`).
  - `CHANNEL_DRAG`.
  - Corrida acoplada con SIS2 por el acoplador FMS.

## Por qué

El benchmark NWA12 (proyecto 4) necesita que pyOCEAN corra un caso regional real de punta a punta. Hoy no puede, por cinco huecos:
1. **No hay mallas curvilíneas.** Las métricas ya son campos (`dx_u`, `dy_v`, `dx_t`, `dy_t`, áreas; `core/dynamics/sigma_baroclinic.py:1511`), pero solo se construyen mallas cartesianas o esféricas regulares (`core/grid/horizontal.py`). No hay lector del supergrid de MOM6, ni ángulo de la malla para rotar vectores.
2. **No hay adaptador de GLORYS.** Existen HYCOM en z, `.a/.b` y archivo nativo, y la ruta `forcing/ocean_init.initial_state/boundary_data` con `open_boundary_state(t)` en el modo `'radiating'`. GLORYS (NetCDF de Copernicus, niveles z) falta.
3. **Las mareas de la frontera 3D no están conectadas.** `forcing/tides.py` tiene 8 constituyentes (sin MM ni MF) y la marea barotrópica por Flather en los modelos 2D. La frontera 3D (`core/dynamics/open_boundary.py`) no suma la marea al exterior (η, ū).
4. **No hay presión atmosférica en el 3D.** `msl` solo entra al bulk.
5. **No hay ríos como fuente puntual en un caso real**, ni MOM6 corrido en un recorte regional con forzamiento real.

**Meta de Esteban para la cadena (2026-10-31):** en NWA12, menos error que MOM6 contra observaciones independientes, solo con física y numérica generales, nunca afinando para este caso.

## Metas

**A. Mallas curvilíneas ortogonales.** **[decisión de Claude]** El supergrid de MOM6 es la forma canónica de entrada.
1. `core/grid/supergrid.py`, función `grid_from_supergrid(hgrid, topog=None, i0, i1, j0, j1)`. Construye un `xr.Dataset` con las mismas variables que hoy (`dx_u`, `dy_v`, `dx_t`, `dy_t`, `area_*`, `f_q`, máscara) y agrega las métricas que una malla curvilínea necesita y la cartesiana no distinguía (`dy_u`, `dx_v`, `dx_q`, `dy_q`), lat/lon de cada punto y `angle_t` (cos/sin).
   - La regla es la de MOM6 `set_grid_metrics_from_mosaic`: cada métrica es la suma de los dos medios segmentos del supergrid y cada área la suma de las cuatro subáreas.
   - Recorta a una ventana de índices.
   - Las mallas cartesianas y esféricas existentes no cambian (bit a bit).
2. **Auditoría de cada uso de métricas** en dinámica, trazadores, viscosidad, PGF, presión de superficie, Coriolis y fronteras, con el operador de MOM6 como referencia. Donde el código use `dx_u` en lugar de `dx_v` (o similar) porque en las mallas actuales eran iguales, se corrige para la forma general.
   - Por construcción, en mallas cartesianas o esféricas regulares las métricas nuevas coinciden con las de hoy, así que las corridas existentes quedan bit a bit. Si en algún punto no es así, se documenta y se regeneran los goldens a propósito.
3. **Rotación de vectores** al ángulo de la malla, como MOM6: viento, esfuerzo y corriente del bulk, u y v de GLORYS, y u y v de la marea.

**B. Adaptador de GLORYS** (`forcing/sources/glorys.py`) sobre el contrato canónico de océano (`forcing/canonical.py`), con el mismo patrón que `hycom_z.py`.
- **Variables:** `thetao` (θ), `so` (S_P), `uo`, `vo`, `zos`.
- **Datos:** diarios, en niveles z. Los huecos bajo tierra y bajo el fondo se rellenan como `tools/boundary/fill_glorys_nn.sh` (vecino más cercano).
- **Descarga:** `scripts/fetch_glorys.py` con `copernicusmarine` (2.3.0 instalado; la credencial ya existe), recortada a la caja del dominio más un margen.
- **Estado inicial:** sale de la ruta existente `initial_state(eta, T, h_src=...)`, con la reconstrucción en los niveles de origen y el regrid que se hizo para BayIW.
- **Fronteras:** `boundary_data` da el exterior por borde y por tiempo, lineal entre registros diarios. El perfil se guarda en z y se remapea a las capas actuales, como hoy.

**C. Mareas en la frontera 3D.**
1. `forcing/tides.py` gana MM y MF (frecuencias, fase de equilibrio y factores nodales de las mismas tablas OTPS/Foreman ya citadas). Los 8 existentes quedan bit a bit.
2. **Lector de TPXO10:** η y transporte (U, V → u = U/H con la profundidad del modelo, como `boundary_tides` hoy) en las caras abiertas de una malla curvilínea, con la rotación de A3.
3. **Suma al exterior:** en el modo `'radiating'` la marea se suma al exterior (η, ū) de Flather. Los registros diarios de GLORYS no resuelven la marea, así que no hay doble conteo. La marea **no** se suma a la velocidad baroclínica u′ del exterior.
   - Es la regla de MOM6, donde `OBC_TIDE` suma amplitud y fase a SSH y a la velocidad barotrópica del segmento (se verifica en la lectura de `MOM_open_boundary`).
4. **Marea de potencial con SAL escalar:** `body_force`/`sal_beta` ya existen en 2D y se conectan al 3D como término barotrópico, con la misma opción.
   - **[decisión de Claude]** Por defecto se activa cuando hay marea en la frontera, como NWA12 (`TIDES = True`). En MOM6 el SAL de NWA12 se lee en la lectura de referencia; la opción `sal_beta` se elige igual.

**D. Presión atmosférica** (`atm_pressure=False` por defecto, bit a bit).
- **En el interior:** −∇p_a/ρ₀ en el momento como término barotrópico. El `msl` lo da el mismo `AtmosphereProvider`, en t^{n+½}.
- **En la frontera:** el exterior η de Flather lleva la corrección de barómetro inverso, η_ext + η_IB con η_IB = −(p_a − p̄_a)/(ρ₀ g). Aquí p̄_a es la media de p_a sobre el dominio mojado en ese instante. Sin esta corrección la frontera reflejaría la respuesta estática.
- **Paridad con NWA12:** NWA12 la tiene apagada (`MAX_P_SURF = 0`), así que la configuración de paridad la apaga. Encenderla es una opción de física general que se juzga en el proyecto 4 contra mareógrafos (marea meteorológica), no aquí.

**E. Ríos** como fuente de volumen de agua dulce en celdas de costa.
- Usa la ruta de agua dulce real de N2a: fw > 0 entra a la capa superior con S = 0 y T = temperatura de la capa, como MOM6 con `runoff` sin calor propio. Se verifica en la lectura de referencia.
- **Dato:** GloFAS v4 diario, mapeado a la celda de mar más cercana a cada desembocadura con la regla de `tools/rivers` de CEFI, que es la que usa NWA12.
- **[decisión de Claude]** No se usan aforos USGS: serían otro dato que MOM6 no ve, y en el proyecto 4 cambiaría la comparación.

**F. MOM6 en el mismo recorte.**
- **Compilación:** MOM6 + SIS2 con el acoplador FMS (`~/pyOCEAN_refs/FMScoupler`, FMS 2025.02.01) y los componentes nulos de atmósfera y tierra, como NWA12.
- **Por qué no `ocean_only`:** no calcula el bulk y obligaría a darle a MOM6 los flujos de pyOCEAN, lo que quita independencia a la comparación. **[decisión de Claude]**
- **Hielo:** en el periodo elegido no hay hielo en el recorte. SIS2 corre solo para que el bulk esté en el mismo lugar que en NWA12.
- **Entradas:** los archivos se escriben **desde la misma definición de caso** que pyOCEAN: supergrid y topografía recortados, `vgrid_75_2m.nc`, segmentos OBC de GLORYS con `tools/boundary/write_glorys_boundary.py` de CEFI, mareas con `write_tpxo_boundary.py`, ERA5 con `tools/atmos`, GloFAS con `tools/rivers`, y estado inicial con `write_glorys_initial.py`. Se usan las herramientas de CEFI para que el MOM6 sea el NWA12 de verdad y no una traducción nuestra.
- **`MOM_input`:** el de `NWA12.COBALT`, sin BGC (sin COBALT), con la malla del recorte y los segmentos que el recorte tenga.

**G. Definición de caso única** en `validation/nwa12_cut/`, con el patrón de DOME e Ilicak: `case.py` (ventana, periodo, rutas), `build_inputs.py` (pyOCEAN y MOM6, y CROCO en el proyecto 4), `run_pyocean.py`, `run_mom6.sh`, `compare.py` y `README.md`.
- El código general (malla, GLORYS, mareas, presión, ríos) va a la biblioteca. En `validation/` solo queda lo propio del caso.
- Los datos y las salidas van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/`, nunca al repo.

## Decisiones del caso

1. **Recorte [decisión de Claude]:** el Mid-Atlantic Bight y el sur de Nueva Inglaterra, aproximadamente 75.5–68.5 °W y 36.5–41.8 °N, en la malla de NWA12 (~1/12°, del orden de 85 × 70 columnas × 75 niveles).
   - **Por qué este recorte:**
     - Contiene la boya NDBC 44025, ya usada en la columna de N2b.
     - Contiene el arreglo OOI Pioneer en el borde de plataforma (40.1 °N, 70.8 °W), con datos en 2017.
     - Contiene los mareógrafos NOAA de Sandy Hook, Atlantic City, Montauk, Newport y Nantucket, las sondas de fondo eMOLT, las líneas de gliders y la Corriente del Golfo con sus anillos en el borde sureste.
   - **Fronteras:** sur, este y el tramo noreste son abiertos; el oeste y el norte, costa.
   - La ventana exacta de índices la elige `case.py` con dos reglas:
     - cada borde abierto queda a ≥ 10 celdas de 44025 y de Pioneer;
     - ningún borde abierto toca celdas con H ≤ 0, porque el ALE no lo admite.
2. **Periodo [decisión de Claude]:** del 2017-05-01 al 2017-09-01. Mayo es arranque; junio a agosto, verano estratificado. Coincide con la columna ERA5 de N2b y con los datos de Pioneer. Los registros de forzamiento cubren del 2017-04-29 al 2017-09-03.
3. **Configuración de paridad con NWA12** (la que se compara con MOM6 en este proyecto):
   - **Vertical y EOS:** z\* con los 75 niveles de `vgrid_75_2m.nc` y `eos="wright_full"` (θ, S_P).
   - **Superficie:** bulk `'ncar'` (FMS, el de NWA12).
   - **Fronteras:** `'radiating'` con nudging de 3 días hacia adentro y 360 días hacia afuera (los valores de NWA12, pasados como parámetros; los valores por defecto de pyOCEAN no cambian), mareas de 10 constituyentes, ríos GloFAS y sin presión atmosférica.
   - **Mezcla y viscosidad:** KPP (CVMix) + Jackson + Smagorinsky laplaciana. NWA12 usa ePBL, doble difusión y Smagorinsky biarmónica, que pyOCEAN no tiene.
   - **Paso de tiempo:** el estable más grande, medido.
   - **Esa física faltante no es parte de este proyecto.** Este proyecto entrega la cadena; la física que falte se clasifica (b) y se decide en el proyecto 4 con observaciones.
4. **Reservorio de trazadores de MOM6 (9 km):** pyOCEAN usa su nudging de Orlanski más la esponja hacia el exterior, que es otro esquema. Se documenta como clase (c) y no se porta en este proyecto. **[decisión de Claude]**

## No metas

- El benchmark contra observaciones con puntaje (proyecto 4).
- CROCO en el recorte (proyecto 4). Las entradas sí se escriben desde la definición de caso, para que agregarlo sea solo un escritor nuevo.
- Física nueva: ePBL, doble difusión, Smagorinsky biarmónica, reservorio de trazadores de MOM6, hielo marino y BGC.
- Mallas no ortogonales.
- Mallas con polo desplazado o tripolares.

## Anti-metas

- Afinar cualquier parámetro para que el recorte salga mejor (la regla de siempre).
- Una ruta de código propia del caso dentro del núcleo.
- Datos, figuras, npz o logs en el repo.
- Cambiar los resultados de las corridas cartesianas existentes sin decirlo: goldens y 9 hashes de smoke bit a bit, salvo la regeneración documentada de A2.

## Criterios de éxito

1. **Malla:** las métricas, las áreas, f y el ángulo del recorte son iguales a las de MOM6 compilado (`ocean_static.nc` de la corrida F en el mismo recorte) a ≤ 1e-12 relativo.
2. **Invariancia en mallas curvilíneas:** en una malla curvilínea analítica (rotada y estirada), el resultado es el mismo que en la malla cartesiana equivalente a ≤ 1e-10 relativo para estos casos:
   - onda de gravedad de superficie libre;
   - ajuste geostrófico;
   - advección de un trazador;
   - un caso de reposo estratificado.
3. **GLORYS:**
   - La lectura y el recorte reproducen `copernicusmarine` punto a punto.
   - La rotación de vectores es exacta en un campo sintético rotado.
   - El estado inicial en capas conserva el contenido de calor y sal de cada columna al remapeo (≤ 1e-12 relativo).
   - El exterior de cada borde cae en los puntos correctos de la malla C.
4. **Mareas solas** (30 días, homogéneo, TPXO10 en la frontera, potencial + SAL): análisis armónico en los mareógrafos NOAA del recorte contra las constantes armónicas de NOAA CO-OPS y TICON-3.
   - Se compara con MOM6 solo con mareas en el mismo recorte y con TPXO10.
   - **Criterio:** el error vectorial medio de M2 de pyOCEAN no es peor que el de MOM6 en el mismo recorte, o la diferencia queda explicada (a)–(d). No se ajusta el arrastre.
5. **Presión atmosférica:**
   - Una prueba estática con p_a uniforme en el tiempo y gradiente constante, en una cuenca cerrada, da η = −p_a′/(ρ₀ g) a ≤ 1 % tras el ajuste.
   - El volumen se conserva exacto.
   - Con la opción apagada, todo queda bit a bit.
6. **Corrida completa de paridad** (4 meses):
   - Termina sin truncamientos de CFL y sin eventos del limitador fuera de lo diagnosticado.
   - Los presupuestos de volumen, calor y sal del dominio, con flujos por la frontera, el bulk y los ríos, cierran a ≤ 1e-10 relativo.
   - El restart a la mitad es bit a bit, con GLORYS, mareas, atmósfera y ríos variando en el tiempo.
   - Las corridas con 1 y con 5 hilos son bit a bit.
   - Se mide el costo por día simulado, junto al de MOM6 en los mismos 5 núcleos P.
7. **MOM6 en el recorte:**
   - Corre los 4 meses con las mismas entradas.
   - El gradiente de presión inicial es igual al de pyOCEAN a ≤ 1e-11 m s⁻² (misma malla, mismo estado inicial y misma EOS; prueba de que la entrada es la misma).
   - Se hace una primera mirada a SST contra OISST y a η en mareógrafos, sin puntaje: eso es del proyecto 4.
8. **Sin cambios en lo existente:** la suite completa pasa, y los goldens y los 9 hashes de smoke quedan bit a bit (salvo A2 documentado).

Cada criterio no cumplido se clasifica (a)–(d), como siempre. Al final va la lista de "qué debe mejorar pyOCEAN" y se actualiza el scorecard.

## Descargas que necesita (requieren la autorización de Esteban)

| Qué | De dónde | Tamaño aproximado | Destino |
|-----|----------|-------------------|---------|
| `ocean_hgrid.nc`, `ocean_topog.nc` y `ocean_mask.nc` de NWA12 | portal de datos de CEFI (GFDL/NOAA PSL) | ~0.5 GB | `~/pyOCEAN_refs/nwa12_grid/` |
| GLORYS12 diario (`GLOBAL_MULTIYEAR_PHY_001_030`), caja del recorte + 1°, 2017-04-29 → 09-03 | Copernicus Marine (`copernicusmarine`, credencial existente) | ~1–2 GB | `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/data/glorys/` |
| ERA5 horario, mismas caja y fechas, con `lsm` | CDS (`~/.cdsapirc` existente, `fetch_era5.py`) | ~1 GB | `.../nwa12_cut/data/era5/` |
| GloFAS v4 diario de 2017 | CDS / EWDS | ~1–3 GB global, se recorta | `.../nwa12_cut/data/glofas/` |
| SIS2 y componentes nulos (atmos_null, land_null) en las versiones de CEFI | GitHub NOAA-GFDL | código | `~/pyOCEAN_refs/` |
| Constantes armónicas NOAA CO-OPS de los mareógrafos del recorte | API de CO-OPS | KB | `.../nwa12_cut/data/tides/` |

TPXO10 ya está en local (`~/pyOCEAN_refs/TPXO10_atlas_v2_nc`), igual que `vgrid_75_2m.nc`.

## Lo que queda al constructor

- El orden de tareas.
- Los nombres internos.
- Las pruebas de cada pieza, más allá de los criterios.
- La forma de los escritores de entrada de MOM6. Las herramientas de CEFI se usan tal cual cuando se pueda; si una necesita un parche (rutas o recorte), el parche va en `validation/nwa12_cut/` y queda documentado.
