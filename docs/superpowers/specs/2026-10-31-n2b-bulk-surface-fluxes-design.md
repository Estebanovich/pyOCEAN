# N2b núcleo: flujos de superficie por fórmulas bulk en el modelo 3D

**Fecha:** 2026-10-31. **Estado:** aprobado por Esteban el 2026-10-31 (documento completo, tras la revisión de constructor).

**Contexto:** `main` en `d44c2b1` (1886 tests). Es el primero de cuatro proyectos acordados para cerrar N2 (`docs/design/vision_roadmap.md` §8, N2; TEOS-10 agregado como proyecto propio el 2026-10-31 tras la revisión de constructor):
1. **Este spec, el núcleo N2b:** fórmulas bulk con ERA5 en el paso 3D, más lo menor que N2a dejó abierto.
2. **TEOS-10 en el 3D:** componente EOS, gradiente de presión, N² de los cierres, restart y goldens. Hoy el 3D solo tiene `LinearEOS` (`core/dynamics/sigma_baroclinic.py:1239`).
3. **La cadena regional:** fronteras abiertas con T, S, u y v variables en el tiempo desde GLORYS, mareas en la frontera en 3D, malla y batimetría del recorte, y MOM6 corrido en el mismo recorte.
4. **El benchmark regional:** un recorte de MOM6 NWA12 en la plataforma noreste de EE. UU., juzgado contra observaciones independientes, con **tres modelos** en el mismo recorte y con el mismo forzamiento: pyOCEAN, MOM6 y CROCO (decisión de Esteban, 2026-10-31). Si dos modelos fallan en el mismo lugar, apunta al forzamiento o a las fronteras; si falla uno solo, a ese modelo.

Lecturas de referencia en `docs/design/n2b_refs/`:
- `croco_bulk.md`: `bulk_flux.F` de CROCO (git ab63b58) línea por línea, con pseudocódigo y `archivo:línea`. Camino por defecto: COARE 3.0 con `BULK_GUSTINESS` y `BULK_LW`, 3 iteraciones, todo en doble precisión. Peculiaridades que el port copia literalmente: `CtoK` 273.16, q_sat con presión constante 1e5 Pa, Charnock fijado con el viento inicial, ψ_t(0) = −0.0045, u\* de la última iteración no recalculado, corrección de Webb, onda larga saliente 0.985 σT⁴ sin reflexión. **Diferencias de acoplamiento de CROCO** (no del algoritmo): sin viento relativo a la corriente, SST = capa superior, E − P como flujo de sal virtual (no volumen), onda corta penetrando con Jerlov I fijo.
- `mom6_fms_bulk.md`: cómo MOM6 y el acoplador de FMS calculan los flujos bulk, la SST de superficie, la radiación y el agua dulce, con `archivo:línea`, y en qué difiere `forcing/bulk.py::bulk_ncar`. **La rutina bulk de FMS (`surface_flux_mod::ncar_ocean_fluxes`, en FMScoupler) no está en disco**: los submódulos de `MOM6-examples/src` están vacíos y FMS 2025.02.01 no la incluye. Lo leído de memoria va [verificar]. Hechos verificados que este spec usa:
  - SST y SSS para el acoplador: promedio ponderado por espesor sobre los primeros `HMIX_SFC_PROP` = 1 m (`MOM.F90:2695, 4242-4301`); velocidad de superficie: capa 1 (`HMIX_UV_SFC_PROP` = 0, `MOM.F90:2700, 4360-4367`);
  - OM4: `ncar_ocean_flux`, `raoult_sat_vap` (factor 0.98), todas las variables atmosféricas a 10 m en `data_table` (OM4 usa JRA55-do; con ERA5 la temperatura y la humedad van a 2 m);
  - constantes de FMS: `CP_AIR` 1004.64 (pyOCEAN 1000.5), `HLV` 2.5e6, `RDGAS` 287.04;
  - agua dulce: flujo de masa real que entra a la T de la capa superior con S = 0 (`MOM_diabatic_aux.F90:1020-1043`), como N2a;
  - interpolación temporal lineal entre registros instantáneos; los acumulados se convierten antes (`time_interp_external2.inc:140-230`);
  - onda corta en OM4: 4 bandas, penetración `MANIZZA_05` con clorofila (`MOM_opacity.F90:352-380`); OM4 restaura salinidad (`RESTORE_SALINITY`, 0.1667 m/día).

## Por qué

pyOCEAN ya tiene las piezas de forzamiento atmosférico, pero sueltas:
- **Fórmulas bulk:** `forcing/bulk.py` trae NCAR (Large & Yeager 2004, el protocolo de MITgcm y CESM) y Kara (HYCOM `flxflg=6`, igual a HYCOM compilado a 1e-10). `compute_surface_fluxes` devuelve τ, Q_net, Q_sw y E − P a partir de una atmósfera canónica y una SST.
- **Datos:** hay adaptadores de ERA5 y WRF al contrato canónico (`forcing/canonical.py`) e interpolación espacial y temporal (`forcing/interp.py`).
- **Modelo 3D:** acepta en cada paso un esfuerzo del viento, un flujo de calor prescrito (`heat`, N3) y E − P real (`fw`, N2a).

Lo que falta es conectarlas. La SST del modelo nunca retroalimenta los flujos, así que una corrida regional realista no se puede hacer sin preprocesar flujos fijos fuera del modelo. El caso regional que cierra N2 (proyecto 4) lo necesita, y también las corridas propias de Esteban con WRF de IOA.

**Meta de Esteban para la cadena completa (2026-10-31):** en el benchmark regional, pyOCEAN debe tener **menos error que MOM6 contra las observaciones**, no solo empatar. Lo logra solo con mejoras generales de física o numérica, nunca ajustando a ese caso. Este spec no se juzga contra esa meta, pero sus decisiones no deben impedirla. En particular, el forzamiento debe poder ser idéntico al de MOM6.

## Metas

**A. Flujos bulk acoplados en el paso 3D** (σ y ALE: σ, z\* e híbrido):
- En cada paso, la atmósfera (canónica, ya en los puntos T del modelo) se interpola en el tiempo.
- Se calculan esfuerzo del viento, calor sensible y latente, evaporación, onda larga neta y onda corta neta con la SST y la corriente de superficie del propio modelo.
- Entran por las vías que ya existen:
  - τ a las caras u y v;
  - Q_ns = Q_lw + Q_sens + Q_lat al calor de superficie;
  - Q_sw con el perfil de Jerlov;
  - P − E como agua dulce real.
- Esquemas:
  - `'ncar'` por defecto, en la forma del acoplador de FMS que usa MOM6;
  - `'coare'`, port del `bulk_flux.F` de CROCO (git ab63b58) con sus opciones por defecto (`BULK_GUSTINESS`, `BULK_LW`). CROCO no trae NCAR, así que pyOCEAN corre con NCAR al compararse con MOM6 y con COARE al compararse con CROCO; la comparación queda entre modelos y no entre fórmulas;
  - `'kara'`, la de HYCOM, que ya existe.

**B. Proveedor de atmósfera** independiente de la fuente:
- un objeto que, dado el tiempo del modelo, devuelve los campos canónicos en los puntos T;
- un constructor para ERA5 a partir de `ERA5Source` y otro para WRF a partir de `WRFSource`;
- la lectura y la interpolación espacial se hacen una vez por registro, nunca en cada paso.

**C. Lo menor que N2a dejó abierto:**
- los acumuladores de TEF (`TEFTransect`, `TEFRegion`) en el restart, bit a bit;
- la viscosidad horizontal de respaldo para AB3 + WENO5 (ver "Decisiones de Claude" 4).

**D. Validación del núcleo** sin el caso regional: algoritmo, conservación, restart, costo y una corrida de columna contra un modelo de referencia con la misma fórmula bulk.

## No metas

- **TEOS-10:** es el proyecto 2. N2b corre con la ecuación de estado lineal y temperatura potencial. La SST que entra al bulk pasa por un solo punto del código, para que TEOS-10 agregue ahí la conversión de temperatura conservativa a potencial, como MOM6 (`ocean_model_MOM.F90:870-897`), sin tocar el bulk.
- **Fronteras abiertas variables, ríos, mareas 3D y el recorte de NWA12:** son el proyecto 3.
- **Comparar contra observaciones regionales:** es el proyecto 4.
- **Usar COARE para mejorar el benchmark contra MOM6.** COARE entra solo para igualar el forzamiento de CROCO. Cambiar la fórmula frente a MOM6 cambia el forzamiento y no el modelo, así que no cuenta como mejora.
- **Hielo marino y flujos de calor por congelamiento**, salvo que la lectura de MOM6 muestre que hace falta un límite de punto de congelación para no dar T < T_f. En ese caso, solo el límite, documentado.
- **Descargar más productos atmosféricos** que ERA5 y WRF.

## Antimetas

Aprobadas por Esteban el 2026-10-31. Es un fracaso aunque el código corra si:

1. **La SST está mal definida.** El bulk usa la capa superior tal cual, aunque mida 1 mm o 200 m. La SST, la SSS y la corriente de superficie deben ser promedios sobre un espesor fijo, como MOM6 (ver `mom6_fms_bulk.md`), y dar lo mismo en z\*, σ e híbrido para la misma columna.
2. **El calor o el agua dulce no cierran.** El calor que entra por la superficie debe aparecer en el contenido de calor a ≤ 1e-13 relativo del cambio, como el flujo prescrito de N3. Lo mismo para el volumen y la sal con E − P. El balance de varianza (`variance_budget`) sigue cerrando.
3. **Unidades o signos de ERA5 equivocados en silencio.** Por ejemplo los acumulados en J m⁻² por hora, la precipitación en m, o las convenciones de signo. Cada conversión se prueba contra un valor de referencia conocido.
4. **Parámetros escondidos o calibrados.** Albedo, emisividad, correcciones o factores elegidos para que el resultado se vea bien. Todos los valores por defecto se citan de NCAR, MOM6/FMS o la fuente.
5. **Restauración que tapa errores.** No hay restauración de SST ni de SSS por defecto. Si se agrega, va apagada, y en el benchmark se usa idéntica a la de MOM6 y se reporta.
6. **Romper lo que existe.** Sin flujos bulk todo queda bit a bit: goldens, smoke 9/9 y benchmarks.
7. **Un restart que no es exacto.** Con flujos bulk, cortar y reanudar debe dar bit a bit, incluida la interpolación temporal de la atmósfera y los acumuladores de TEF.
8. **Costo escondido.** El bulk completo (interpolación temporal, cálculo y paso a caras) cuesta ≤ 5 % por paso en BayIW cut z\*. Leer archivos dentro del paso cuenta en ese costo.
9. **Flujos en tierra o en celdas secas.** Una columna seca o de tierra no recibe calor, agua ni esfuerzo, y queda quieta bit a bit, como con el flujo prescrito.

## Criterios de éxito

1. **Algoritmo bulk igual a la referencia.**
   - Con la misma SST, la misma corriente y la misma atmósfera, los flujos de `'ncar'` coinciden con la rutina NCAR del acoplador de FMS que usa MOM6. La tolerancia es 1e-10 relativo, o la que el constructor justifique, sobre un conjunto de puntos que cubra estable, inestable, viento débil y fuerte, y SST alta y baja.
   - El constructor trae el código público de FMScoupler (GitHub NOAA-GFDL, la versión que corresponde a FMS 2025.02.01) a `~/pyOCEAN_refs/`, completa `mom6_fms_bulk.md` quitando los [verificar] y compila un driver mínimo de `ncar_ocean_fluxes`.
   - Si `bulk_ncar` difiere de FMS (alturas, iteraciones, factor de humedad, densidad del aire, ráfagas), cada diferencia se clasifica. Por defecto, pyOCEAN adopta la forma de FMS/MOM6 para que el proyecto 4 compare modelos y no fórmulas, y conserva la de MITgcm como opción solo si la diferencia es una elección (c).
   - Con la misma prueba, `'coare'` coincide con `bulk_flux.F` de CROCO compilado a 1e-10 relativo o la tolerancia justificada. El driver es un `bulk_point` mínimo con el bloque escalar de `bulk_flux.F:199-659` y las funciones de `:758-972`, que no dependen de MPI ni de bloques comunes. Las entradas son las de CROCO: viento absoluto y presión de 1e5 Pa.
   - También se compara `'ncar'` con `pkg/exf` de MITgcm (pendiente desde la fase 2 del build), a la tolerancia que la diferencia de algoritmo permita.
2. **Conversión de ERA5.**
   - Un archivo ERA5 real (horario: u10, v10, t2m, d2m, msl, ssrd, strd, tp; el constructor lo descarga con la API de CDS) pasa por el adaptador y da flujos canónicos.
   - Las conversiones (acumulado → tasa, punto de rocío → humedad específica, m → m s⁻¹) se comprueban contra valores calculados a mano en al menos un punto y una hora.
   - Promediada en un mes, la radiación coincide con los campos medios de flujo de ERA5 (`msdwswrf` y similares) donde estén disponibles, o con su media.
3. **Conservación.**
   - Calor: Δ(Σ ρ₀ c_p T V) = Σ Q A Δt a ≤ 1e-13 relativo del cambio.
   - Volumen y sal: Σ ΔV = Σ (P − E) A Δt al nivel de N2a, y la sal total se conserva.
   - Se cumple en σ, z\* e híbrido, con mojado y secado y con columnas de tierra.
4. **SST independiente de la coordenada.** Una columna en reposo, horizontalmente uniforme y forzada igual da la misma SST, los mismos flujos y el mismo T(z) en z\* (capas fijas) y en σ a ≤ 1e-9. La híbrida, con su capa superior de distinto espesor, da los mismos flujos a la tolerancia que permita el remapeo, justificada.
5. **Columna contra un modelo de referencia con la misma fórmula bulk.**
   - Una columna (5×5 periodica, como `validation/turbulence_columns/`) forzada por ERA5 horario en un punto de la plataforma noreste de EE. UU., 30 días o más.
   - Dos referencias, cada una con su propia fórmula:
     - con `'ncar'`: MOM6 si su configuración de columna admite el bulk de FMS; si no, MITgcm con `pkg/exf` (la misma LY04). El constructor elige y lo justifica;
     - con `'coare'`: CROCO con `BULK_FLUX`, compilado como los casos de `validation/turbulence_columns/`.
   - Los tres con KPP, que pyOCEAN, MOM6/CVMix y CROCO ya tienen.
   - Las diferencias de acoplamiento de CROCO (viento absoluto, SST de la capa superior, sal virtual, Jerlov I) son elecciones de CROCO, clase (c) o (d). La corrida de pyOCEAN con `'coare'` se repite una vez con opciones que las imitan (viento absoluto, `h_sfc` = capa superior) para separarlas del resto. Las opciones existen solo para esa separación y no cambian los valores por defecto.
   - Se reportan SST, profundidad de la capa de mezcla y los flujos, con cada diferencia clasificada de (a) a (d).
   - Criterio: flujos medios dentro del 2 % y SST dentro de 0.2 °C al final, o la diferencia explicada.
6. **Nada se rompe.**
   - Sin bulk todo queda bit a bit (goldens y smoke 9/9 contra `main`).
   - Restart bit a bit con bulk, en σ e híbrido.
   - 1 = 5 hilos.
   - El pre-flight rechaza juntas todas las combinaciones inválidas: bulk y `heat`/`fw`/`tau` prescritos a la vez, bulk sin `salinity` (E − P lo exige), un esquema desconocido, `'kara'` sin SSS, `'coare'` sin los campos que pida (p. ej. la onda larga si `BULK_LW`), o una atmósfera que no cubre el periodo de la corrida.
7. **TEF en el restart:** una corrida de Lorenz partida con restart da los mismos acumuladores TEF que la corrida continua, bit a bit.
8. **Costo:** ≤ 5 % por paso en BayIW cut z\* con bulk ERA5 contra el mismo caso con los mismos flujos prescritos, medido como `scripts/bbl_cost.py` (alternado, 3 rondas).
9. **Documentación:** manual (teoría y guía de configuración), `CLAUDE.md`, marcador §7 y hoja de ruta §8 de `vision_roadmap.md`, y `reading_list.md`.

## Decisiones de diseño

Aprobadas en la conversación del 2026-10-31.

- **ERA5 por defecto y WRF como segunda fuente.** Todo pasa por el contrato canónico, así que el paso 3D no sabe de qué fuente viene la atmósfera.
- **NCAR (LY04) por defecto**, en la forma del acoplador de FMS que usa MOM6 [verificar en `mom6_fms_bulk.md`]. COARE como port de CROCO para comparar con CROCO (decisión de Esteban: CROCO es el tercer modelo). Kara queda como opción.
- **Acoplamiento en cada paso:** τ, el calor latente y sensible y la onda larga que sale dependen de la SST y la corriente del modelo. Viento relativo a la corriente de superficie, como LY04 y como `bulk_ncar` hoy.
- **Onda corta** con el perfil de Jerlov de N3 (`physics/surface/shortwave.py`) y el remanente del fondo repartido como MOM6. OM4 usa 4 bandas y `MANIZZA_05` con clorofila: diferencia (c), documentada. Si la configuración de NWA12 lo usa, `MANIZZA_05` entra como opción en el proyecto 3, no aquí.
- **Alturas de referencia** configurables por fuente (`z_u`, `z_t`, `z_q`): ERA5 da el viento a 10 m y la temperatura y la humedad a 2 m, y `bulk_ncar` ya corrige el perfil. En el proyecto 3 se usan las alturas que declare la configuración de NWA12.
- **Restauración de salinidad** como flujo de sal (forma de MOM6, `RESTORE_SALINITY`), apagada por defecto. Entra en el proyecto 3 solo si NWA12 la usa [verificar], para usarla idéntica.
- **E − P** entra por la vía de agua dulce real de N2a. El calor que lleva la masa de lluvia y evaporación (a qué temperatura entra y sale el agua) se trata como MOM6 [verificar].
- **Sin restauración** de SST ni SSS por defecto.

## Decisiones de Claude (a revisar por Esteban)

1. **Interfaz.** `SigmaBaroclinic(..., surface_fluxes=BulkFluxes(atmosphere, scheme='ncar', ...))`, con `step(st)` sin `tau`/`heat`/`fw`. La atmósfera es un proveedor `atmosphere(t) -> dict` en el tiempo del modelo [s desde el origen de la corrida], con fecha de origen explícita, como `open_boundary_state(t)`. El componente de flujos de superficie lo llama el controlador antes de la fase que hoy recibe `tau` (fase 1/2) y deja `heat` y `fw` para las fases que ya los usan.
2. **En qué tiempo se evalúa.** La atmósfera se interpola linealmente en t^{n+½} (punto medio del paso). La SST y la corriente son las del inicio del paso (estado n), como el acoplamiento explícito de MOM6 en un paso del acoplador [verificar]. Todo el paso usa los mismos flujos.
3. **SST, SSS y corriente de superficie.** Son promedios ponderados por espesor sobre los primeros `h_sfc` = 1 m (el `HMIX_SFC_PROP` de MOM6), calculados igual en σ, z\* e híbrido. Donde la columna es más somera que `h_sfc`, el promedio es sobre toda la columna. MOM6 usa para la corriente la capa 1 (`HMIX_UV_SFC_PROP` = 0); pyOCEAN usa el mismo promedio de 1 m también para la corriente, por la antimeta 1 (una capa superior de 1 mm o de 200 m no debe decidir el viento relativo). Es una diferencia (c), pequeña y documentada; `h_uv_sfc` = 0 la reproduce como opción para el proyecto 4.
4. **Viscosidad de respaldo para AB3 + WENO5.** Desde el 2026-10-30 Smagorinsky existe en σ y ALE (`smagorinsky=C`). La prueba de Lorenz D σ + GLS a T_M2/498 completó 150 ciclos con y sin él, así que el respaldo no extiende el Δt estable. Propuesta: cerrar el punto con una prueba de regresión (Lorenz D σ + GLS a T_M2/498 con C = 0.15, 150 ciclos, ya medido) y la documentación, sin código nuevo. Si Esteban quiere un respaldo por defecto (por ejemplo Smagorinsky activo en σ con AB3), es un cambio de comportamiento por defecto que rompe bit a bit y va en un spec aparte.
5. **Lugar del código.**
   - `core/components/surface_fluxes.py`: el componente, el promedio de superficie y el paso a caras.
   - `forcing/atmosphere.py`: el proveedor y sus constructores ERA5/WRF.
   - `forcing/bulk.py`: los algoritmos.
   - Registro de variables: `q_sens`, `q_lat`, `q_lw`, `q_sw`, `evap`, `precip`, `tau_x`, `tau_y`, `sst_bulk` (salida CF/Zarr).
6. **Reproducibilidad del proveedor.** El proveedor es una función pura del tiempo (sin estado que avance), así que el restart no necesita guardarlo. El restart guarda la fecha de origen y la comprueba.
7. **Esfuerzo en las caras.** τ se calcula en los puntos T y se pasa a las caras u y v con el promedio de los dos puntos vecinos, como MOM6 [verificar la forma exacta, que puede promediar u\*² o τ].
8. **Radiación de `'ncar'`** (revisión de constructor). Por defecto, los valores de LY04 que `forcing/bulk.py` ya cita (albedo 0.066, ε = 0.97 en la onda larga que sale y en la reflexión), los del protocolo NCAR/CORE y MITgcm. FMS solo calcula los flujos turbulentos; en OM4 el albedo y la onda larga los pone SIS2 (`ocean_albedo_option = 2`), cuyo código no está en disco. Igualar la radiación de NWA12 es trabajo del proyecto 3, al leer su configuración. `'coare'` usa la convención de CROCO dentro del port (onda corta neta de entrada, 0.985 σT⁴ sin reflexión).
9. **Bulk y forzamiento prescrito a la vez** (revisión de constructor). En N2b se rechazan juntos, como dice el criterio 6. Mezclar (por ejemplo, esfuerzo de WRF con calor bulk) queda para cuando un caso lo pida.
10. **Presión atmosférica** (∇p_a en el momento y la superficie libre, barómetro inverso; revisión de constructor). No entra en N2b: es un término nuevo de la dinámica en σ y ALE. Va en el proyecto 3, porque el nivel del mar del proyecto 4 lo necesita para igualar a MOM6. El proveedor ya lleva `msl`.
11. **Documentación de TEOS-10.** La fila N2a de `CLAUDE.md` dice que TEOS-10 está conectado en el 3D, pero el 3D solo construye `LinearEOS` (`core/dynamics/sigma_baroclinic.py:1239`, `core/components/eos.py:4`). Se corrige en este proyecto.

## Lo que queda para el constructor

- La forma del componente dentro del contrato de N1, el orden exacto dentro de las fases y el reparto de las pruebas.
- El caché de registros de la atmósfera (ventana de dos registros).
- El conjunto de puntos para el criterio 1 y la herramienta para correr la rutina de FMS (un driver Fortran mínimo como `validation/mom6_column/`).
- La elección del modelo de referencia de la columna (criterio 5) y el punto ERA5.
- Las figuras.

## Restricciones

- Solo Python.
- ≤ 5 hilos en P-cores, con `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`.
- Nunca `pip install`; numpy 1.26.4.
- Commits con rutas explícitas.
- Salidas y datos descargados fuera del repo (`/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`).
- Literatura: primero la biblioteca local y luego Consensus, con 1–2 búsquedas como máximo. Los papers usados van a `docs/literature/reading_list.md`.
- Leer el código de MOM6/FMS antes de implementar (`docs/design/n2b_refs/`).
- Consultar a Esteban si una corrida pasa de ~24 h.
- No ajustar ningún parámetro a un caso.
