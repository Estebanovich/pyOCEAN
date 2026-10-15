# Benchmark DOME: overflow en 3-D con rotación (cierre de la etapa C)

**Fecha:** 2026-10-14. **Estado:** diseño aprobado por Esteban por secciones (2026-10-14); falta revisar el texto escrito.
**Rama:** `dome-benchmark`, desde `main` (0481616).

## Motivo

La etapa C del roadmap (`docs/design/vision_roadmap.md` §8) cierra con el overflow DOME y el modo barotrópico split-explicit. DOME es el primer caso 3-D con rotación, talud y una entrada de agua densa. Responde dos preguntas:

1. ¿pyOCEAN baja y mezcla la pluma como los modelos de referencia, en z\*, σ e híbrida?
2. ¿Aparecen en un overflow realista los pendientes de capas delgadas de la híbrida (README de Ilicak, 4b y 4d)? Si no aparecen, siguen en espera; si aparecen, entran con evidencia.

La forma de trabajo es la que más ha rendido: un caso idealizado barato, una sola definición para todos los modelos, lectura del código de las referencias y clasificación de cada diferencia en (a) error nuestro, (b) física que falta, (c) elección numérica o (d) error de la referencia.

## Referencias

- **Legg, Hallberg & Girton (2006)**, Ocean Modell. 11, 69–97, [10.1016/j.ocemod.2004.11.006](https://doi.org/10.1016/j.ocemod.2004.11.006): define el caso (§2, ec. 3–8, tablas 1–3) y las métricas (ec. 11–13, 18; Figs. 15–17). Es la definición canónica. Biblioteca local: `Legg_2006 - ….md`.
- **Ezer & Mellor (2004)**, Ocean Modell. 6, 379–403: la variante original de DOME (5 Sv). **No** es la que usamos; Burchard & Rennau (2008) la siguen.
- **Ezer (2004)**, Ocean Modell. 9, 151–168: entrainment frente a MITgcm no hidrostático.
- **MOM6** `src/user/DOME_initialization.F90` (MOM6 en `~/pyOCEAN_refs/MOM6`) y `MOM6-examples/ocean_only/DOME/MOM_input`: la implementación de Hallberg del Caso 1 de Legg. Su transporte de entrada, tr₀ = D·√(D g′)·½·L_ρ = 8.55 × 10⁶ m³/s (`DOME_initialization.F90:402-405`), es el de Legg, T_in = Δb₀h₀²/(2f).
- **MITgcm** `verification/dome` es otra variante (2000 × 600 km, dy variable, borde oeste abierto). Solo la usamos como plantilla de código.

## Decisiones tomadas con Esteban (2026-10-14)

1. **Referencias:** MOM6 ρ (isopicnal nativo) y MITgcm z; además los resultados publicados de Legg 2006. MOM6 z\* no, porque su entrada está definida por índice de capa y en z\* metería el agua de la "capa 25" en la capa z número 25 (`DOME_initialization.F90:430-492`).
2. **Resoluciones:** 10 km y 5 km.
3. **Trazador pasivo genérico** en pyOCEAN, no S con β = 0.
4. **Entrada:** borde norte abierto en modo `clamped`. Fuera del canal el borde es tierra, así que solo el canal recibe agua.
5. **Esponja interior general**, independiente de las fronteras abiertas.
6. **Fricción de fondo cuadrática en ALE.** Hoy no existe: ALE exige `z0=None` (`sigma_baroclinic.py:1148-1150`), así que r = 0. Por defecto queda apagada en este plan.
7. **Física común:** laplaciana constante y sin Jackson, con los valores del MITgcm de Legg a 10 km.

## Alcance

**Dentro:**
- Tres capacidades nuevas en pyOCEAN: trazadores pasivos (§1), esponja interior (§2) y fricción cuadrática en ALE (§3).
- Flujo de entrada exacto en un borde `clamped` (§4).
- El caso DOME para pyOCEAN, MOM6 y MITgcm, con las corridas y las métricas (§5–§7).

**Fuera** (se anotan como pendientes en el README):
- Smagorinsky o viscosidad biarmónica (b).
- Mezcla por cizalla tipo Jackson o Hallberg (2000) (b).
- Encender la fricción por defecto en ALE: es una decisión aparte, porque cambia todas las validaciones.
- Los casos 2–6 de Legg.
- La resolución de 2.5 km.
- MOM6 z\*.
- CROCO.

## Criterios de éxito

1. **MITgcm reproduce a Legg** a 10 km, en la configuración de Legg (§6.3): trayectoria Y(x) y flotabilidad final como en sus Figs. 15–17, leídas de la figura y con la tolerancia que permita leerla (~10 %). Esto valida nuestra configuración del caso.
2. **pyOCEAN z\* ≈ MITgcm z** con la física común: Y(x), T(x) y flotabilidad final a ±15 %.
3. **La híbrida de pyOCEAN ≈ MOM6 ρ** con la misma ν_v, en las mismas métricas y a ±15 %.
4. **La tendencia con la resolución** (10 → 5 km) va en la dirección de Legg, o la diferencia queda explicada.
5. **T y s en rango exacto** (s ∈ [0, 1], T dentro del rango inicial y de la entrada). Ninguna corrida se detiene: sin truncaciones CFL fuera de lo normal, `n_limited` reportado.
6. **El transporte de entrada** medido en pyOCEAN es el prescrito a ±1 %, y por clase de densidad coincide con el de MOM6 a ±5 %.
7. **Cada diferencia clasificada** como (a)–(d). El README termina con la lista de "qué debe mejorar pyOCEAN" y el marcador §7 del roadmap queda actualizado.
8. **Costo** al Δt máximo estable, 1 proceso, para los tres modelos.
9. **Las validaciones existentes no cambian:** con `n_tracers=0`, sin esponja interior y sin fricción, la suite y las huellas quedan bit a bit igual.

## §1. Trazadores pasivos (`n_tracers`, solo en ALE)

- **Estado:** `SigmaState.tr` con forma (ntr, N, ny, nx), `None` si `n_tracers == 0`. `initial_state(..., tr=None)`, que pone 0 por defecto.
- **Advección:** los trazadores se agregan al stack que ya se advecta (`stack = [T, T², tr…]`, `sigma_baroclinic.py:2040`). Así usan WENO5 + el FCT de Zalesak con sus cotas 3-D propias por trazador, y el mismo flujo de grosor. Los kernels ya son genéricos en `ntr` (`_advect_faces_h`, `_advect_div`, `_fct_*`, `_advect_fct`, `_advect`). Lo que hoy está fijo en `[T_in, T_in²][:ntr]` (`_open_faces`, `_rhs`, `_ext_stack`) se generaliza a una lista de valores exteriores.
- **Remapeo:** `tr` entra en la lista de campos de `_regrid_remap`. El regrid híbrido sigue usando solo T para la densidad.
- **Difusión:**
  - **vertical:** la misma K_v que T (`kv`), que en este caso es 0;
  - **horizontal:** a lo largo de la capa sin restar ρ_ref (en `_diffuse_h` la anomalía solo aplica a T).
- **Fronteras abiertas:** `OBData.tracers` opcional, con forma (ntr, nz, n) sobre el `zw` del borde, 0 por defecto. En las caras de entrada el valor exterior entra aguas arriba, igual que T. `validate_state` revisa la forma.
- **Esponja de borde y esponja interior:** relajan `tr` hacia su objetivo (§2).
- **Restart:** `tr` y su perfil exterior en `core/restart.py`, bit a bit. `n_tracers` va en la huella de configuración.
- **Salida y diagnósticos:** `tr` en el registro de variables, el contenido de cada trazador y su excursión fuera de [min, max] iniciales.
- **Pruebas** (`tests/test_sigma_ale.py`, `tests/test_restart.py`):
  1. **Oráculo:** con `tr[0] = T` inicial, sin difusión y el mismo exterior, `tr[0] == T` bit a bit después de N pasos, en z\*, σ e híbrida. Es la prueba más fuerte: el mismo camino de código. Si el regrid híbrido usa T, el trazador sigue la misma remapeada.
  2. **Conservación** del contenido en caja cerrada, a redondeo.
  3. **Pulso de 0/1** con flujo cortante: s ∈ [0, 1] exacto.
  4. **Restart bit a bit** con trazador.
  5. **`n_tracers = 0`:** las huellas actuales (`lock_defaults`, `openbox_defaults` y las demás) no cambian.
- **Fuera:** el modo σ no ALE lanza un error claro si `n_tracers > 0`.

## §2. Esponja interior (`sponge_rate`, `sponge_T`, `sponge_tr`)

- **Parámetros:**
  - `sponge_rate`: (ny, nx) [1/s], ≥ 0;
  - `sponge_T`: perfil objetivo con `sponge_zw`, interfaces de arriba hacia abajo [m];
  - `sponge_tr`: objetivos de los trazadores (0 por defecto).
- **Aplicación:** después del regrid-remap de cada paso, implícita como `apply_sponge` de MOM6: T ← (T + Δt·r·T_obj)/(1 + Δt·r), con T_obj remapeado a las capas actuales por el mismo camino que la esponja de borde (`ale.exterior_on_layers`). Lo mismo para cada trazador.
- **No amortigua el momento.** DOME tampoco lo hace (`DOME_initialization.F90:171-252`).
- **Diferencia (c) con MOM6:** su esponja de DOME relaja las alturas de interfaz en modo de capas. En z\*, σ e híbrida relajar T hacia el perfil inicial es el equivalente.
- **Compatibilidad:** coexiste con la esponja de borde (`sponge_cells`); si las dos tocan una celda, se aplican en serie y se documenta.
- **Pruebas:**
  1. Una columna con r constante se relaja exacto como (1 + Δt·r)⁻ⁿ.
  2. Con r = 0 o `sponge_rate=None` el resultado es bit a bit igual.
  3. Con la banda en un borde, el resto del dominio no cambia en un paso, a redondeo.
  4. Restart con esponja interior bit a bit (la tasa y el objetivo van en la huella de configuración).

## §3. Fricción de fondo cuadrática en ALE (`bottom_drag="quadratic"`, `cd`, `drag_bg_vel`, `hbbl`)

- **Fórmula:**

  τ_b = −ρ₀·r·u_b,  r = C_d·√(ū_bbl² + v̄_bbl² + u_bg²)

  - ū_bbl es la velocidad media de la columna de cara en los `hbbl` metros de fondo, ponderada por grosor, como el `HBBL` de MOM6 (`MOM_set_viscosity.F90`). Así una capa de fondo evanescente no decide sola el arrastre.
  - v se interpola a las caras u y u a las caras v, con el promedio de 4 puntos del grid C.
- **Dónde entra:** como esfuerzo implícito en el lecho del paso viscoso acoplado que ya existe (`SigmaWetDry.column_step`, `dz_drag`). La corrección δ de la media vertical ya incluye la fricción implícita (`sigma_wd.py`, cola de `column_step`), así que el balance de la media vertical se conserva.
- **Por defecto apagada** (`bottom_drag=None`), para que todas las huellas y validaciones queden iguales. El log-law sigue disponible en el modo σ no ALE.
- **Pruebas:**
  1. **Decaimiento analítico:** un flujo barotrópico uniforme U₀ en un canal periódico o cerrado y homogéneo decae como U(t) = U₀/(1 + C_d·U₀·t/H). Se compara a tolerancia de discretización temporal; θ implícito da cota de error conocida.
  2. **Capa de fondo evanescente en la híbrida:** estable, y la velocidad de fondo no supera la de las capas de arriba.
  3. **`bottom_drag=None`:** bit a bit igual.
  4. **`u_bg > 0` con U = 0:** sin aceleración espuria.
  5. **Restart** con fricción (parámetros en la huella de configuración).

## §4. Flujo de entrada exacto en bordes `clamped`

- **Problema:** hoy, en un borde `clamped`, la media vertical sigue pasando por Flather: ū = ū_ext + √(g/H)(η − η_ext) (`free_surface.py:36-44`). En DOME entran 8.55 Sv en una caja cerrada, y el nivel medio sube ~35 m en 50 días (3.7 × 10¹³ m³ en 1.05 × 10¹² m²). Con η_ext = 0, Flather cortaría la entrada (√(9.8/600)·35 ≈ 4.5 m/s de salida).
- **Diseño:** `OBData` acepta `prescribed_flux=True`. En ese borde, la cara impone ū = ū_ext exacto (Dirichlet en la superficie libre implícita, como `SIMPLE` de MOM6) y no hay término de radiación.
- **Prueba:** en una caja cerrada con un canal de entrada, el transporte por la cara es el prescrito a 1e-12 relativo en cada paso, y el volumen crece exacto.
- **Alternativa descartada:** η_ext = η interior (Flather degenerado). Depende de la discretización y no garantiza el flujo.

## §5. El caso DOME (`validation/dome/`)

Una sola definición en `validation/dome/cases.py`, como en Ilicak; cada modelo tiene su escritor y su lector.

**Geometría** (la de MOM6, que es la de Legg):
- **Dominio:** 1500 × 700 km. Cartesiano, plano f con f = 1 × 10⁻⁴ s⁻¹ (β = 0).
- **Malla:** dx = dy = 10 km (150 × 70) y 5 km (300 × 140).
- **Batimetría** (`DOME_initialization.F90:97-110`):
  - para y < 600 km: D = min(600 + 0.01·(600 km − y), 3600 m), con fondo plano de 3600 m para y < 300 km y talud de 1 % entre y = 300 y 600 km;
  - para y ≥ 600 km: D = 600 m en el canal (1000 < x < 1100 km) y tierra fuera de él.
- **Paredes:** todas cerradas, salvo el borde norte del canal (entrada).

**Estratificación y EOS:**
- **EOS lineal:** ρ = ρ₀ − α_T(T − T_ref), con ρ₀ = 1031 kg/m³ y g = 9.8 m/s². Valores de MOM6; Legg escribe g′ = 0.019.
- **Ambiente:** N² lineal con Δρ = 2 kg/m³ entre la superficie (1030) y 3600 m (1032), así que Δb₀/(N²h₀) = 12 (tabla 2 de Legg).
- **Entrada:** su densidad máxima es la del fondo del talud, 1032 (Legg §2).
- **En pyOCEAN y MITgcm:** el perfil inicial es la versión continua.
- **En MOM6 ρ:** las 25 capas de `DOME_initialize_thickness`, con R_k = 1030 + (k − 1)·2/24.
- **Diferencia (c):** de discretización vertical; MOM6 discretiza la misma N² en clases.

**Entrada** (Legg ec. 3–8):
- **Parámetros:** h₀ = 300 m, Δb₀ = g·Δρ/ρ₀ = 0.019 m/s², U₀ = √(Δb₀h₀) ≈ 2.39 m/s, L_ρ = U₀/f ≈ 23.9 km, Ri_m = 1/3, h_e = 600 m.
- **Grosor de la capa densa:** h(x_w) = h₀·exp(−x_w/L_ρ), donde x_w es la distancia a la pared derecha mirando aguas abajo. En DOME es la pared oeste del canal, x = 1000 km (`DOME_initialization.F90:450-457`).
- **Coordenada vertical:** en esta sección z es la profundidad, positiva hacia abajo (0 en la superficie, h_e en el fondo del canal). Con esa convención la ec. 4 de Legg da el ambiente b₀ − N²z, que decrece con la profundidad. El texto del PDF perdió los signos al extraerse; la convención queda fijada por estas dos condiciones (ambiente correcto y ζ = −1 en el fondo).
- **Coordenada de la interfaz:** ζ = (h_e − z − h(x_w))/h(x_w). Vale 0 en la interfaz, −1 en el fondo y es positiva arriba.
- **Función de transición** (Legg ec. 8):
  - F = 1 para ζ ≥ Ri_m/(2 − Ri_m);
  - F = ζ/(Ri_m(ζ + 1)) + ½ para −Ri_m/(2 + Ri_m) < ζ < Ri_m/(2 − Ri_m);
  - F = 0 para ζ ≤ −Ri_m/(2 + Ri_m).

  Es continua: vale 1 y 0 en los extremos de la transición.
- **Flotabilidad y velocidad** (Legg ec. 4–5):
  - b(x_w, z) = min(b₀ − Δb₀(1 − F), b₀ − N²z);
  - v(x_w, z) = −U₀·exp(−x_w/L_ρ)·(1 − F), hacia el sur.
- **Prueba de la definición:** el transporte de la ec. 5 integrada es Δb₀h₀²/(2f) cuando la interfaz es delgada, y el transporte por clase de densidad se compara con el de las capas de MOM6 (`DOME_set_OBC_data`, `DOME_initialization.F90:430-446`), a ±5 % (criterio 6). MOM6 usa un perfil logarítmico por capa, no la F de Legg; la diferencia es (c).
- **Transporte total:** ≈ Δb₀h₀²/(2f) = 8.55 Sv. MOM6 da 8.80 Sv con la transición.
- **Trazador:**
  - Legg: s = 1 en el agua de entrada, que es la que tiene F < 1.
  - MOM6: tr_D1 = 1 en las capas k ≥ 12 de la entrada (`DOME_initialization.F90:497-501`).
  - Nuestra regla: s = 1 donde b < b₀ − N²z, es decir, el agua más densa que el ambiente. Es (c) frente a MOM6 y se mide.
- **En pyOCEAN:** el borde norte se abre en modo `clamped` con `prescribed_flux=True` (§4). `OBData` lleva T(z) y v(z) sobre un `zw` fino del borde (≥ 60 niveles en 600 m), y s(z) en `tracers`.

**Esponjas** (`DOME_initialize_sponges`, `DOME_initialization.F90:171-252`):
- **Tasa:** sea d la distancia a la pared este u oeste. La tasa es r_max para d < 100 km, r_max·(200 − d)/100 para 100 km ≤ d < 200 km y 0 después, con r_max = 10 día⁻¹; Idamp = max(oeste, este).
- **Objetivos:** T hacia el perfil inicial y s hacia 0.
- **Momento:** no se amortigua.

**Física común** (MITgcm de Legg a 10 km, tabla 3):
- **Viscosidad horizontal laplaciana:** ν_h = 50 m²/s a 10 km y 25 m²/s a 5 km (∝ Δx, Re de malla constante).
- **Viscosidad vertical:** ν_v = 0.5 m²/s.
- **Difusividad explícita de trazadores:** 0, horizontal y vertical, como Legg en MITgcm.
- **Fricción de fondo:** cuadrática con C_d = 2 × 10⁻³, u_bg = 0 y `hbbl` = 10 m en pyOCEAN y MOM6. MITgcm la aplica en la celda de fondo; es (c) y se documenta.
- **Sin parametrizaciones de mezcla:** sin Jackson y sin convección adicional más allá de lo que haga cada modelo por defecto en su coordenada. Se documenta.
- **Duración:** 50 días. Las métricas se promedian sobre los días 30–50, ajustable si Y(x) no está estacionaria. Salida diaria de T, s, u, v, h y η a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/`.

**Coordenadas de pyOCEAN:**
- **Capas:** 25 en las tres coordenadas.
- **z\*:** 25 niveles uniformes de 144 m, igual que el MITgcm de Legg.
- **σ:** 25 capas uniformes.
- **Híbrida:**
  - objetivos de densidad en las interfaces de las clases de MOM6;
  - piso z\* hasta el fondo, como en el benchmark de Ilicak;
  - `init_iterations` como en Ilicak.
- **Defaults del plan de robustez:** acoplamiento 'mom6', FCT, θ_bc = ½.
- **Δt:** el máximo estable de una escalera (10 km: 1200, 900, 600, 300 s); las métricas se miden con un Δt común si las escaleras difieren.

## §6. Modelos de referencia

### §6.1 MOM6 ρ (`validation/dome/mom6.py`, reutiliza `validation/ilicak_benchmark/run_mom6.sh` y el build `repro`)

- **Configuración:** el `MOM_input` de `ocean_only/DOME` con cambios para la física común:
  - viscosidad: `LAPLACIAN=True`, `KH=50` (25 a 5 km), `BIHARMONIC=False`, `SMAGORINSKY_AH=False`, `AH_VEL_SCALE=0`;
  - mezcla vertical: `KV=0.5`, `KD=0`, `USE_JACKSON_PARAM=False`;
  - fricción: `CDRAG=0.002`, `DRAG_BG_VEL=0`, `HBBL=10`;
  - tiempo: `DAYMAX=50`;
  - malla: `NIGLOBAL`/`NJGLOBAL` por resolución;
  - `LAYOUT = 1, 1`.
- **Variante informativa:** `KV = 1e-4`, que es el HIM de Legg sin su mezcla por Ri; la mezcla por Ri no la tenemos.
- **Salida:** diaria de h, e, u, v y tr_D1. El `diag_table` se escribe desde el escritor.

### §6.2 MITgcm z (`validation/dome/mitgcm.py`, build propio en `~/pyOCEAN_refs/mitgcm_runs/dome/`)

- **Plantilla:** `verification/dome` del código de MITgcm (OBCS norte con perfil prescrito por archivo, `ptracers` para s, `rbcs` para las esponjas).
- **Malla y coordenada:** 150 × 70 o 300 × 140, 25 niveles de 144 m, celdas parciales (`hFacMin` = 0.1), superficie libre no lineal z\* (`nonlinFreeSurf=4`, `select_rStar=2`), por la subida de ~35 m del nivel.
- **Dos configuraciones a 10 km:**
  - **"Legg":** fondo no-slip, como en la ec. 10 de Legg. Sirve para el criterio 1.
  - **"Común":** fondo free-slip + fricción cuadrática. Sirve para el criterio 2.
- **A 5 km:** solo la común.
- **Advección de T y s:** limitador de flujo (`tempAdvScheme=77`, como Legg: Pietrzak 1998).

### §6.3 Anclas publicadas de Legg

Se leen de las Figs. 15–17 del PDF local, para el Caso 1 a 10 km:
- flotabilidad final del agua producto, z (ν_v = 0.5) e isopicnal con ν_v = 0.5 y 1e-4;
- trayectoria media Y(x);
- la observación de que la pluma isopicnal viscosa baja ~40 km más que la no viscosa.

Los valores leídos quedan en `cases.py` con la tolerancia de lectura.

## §7. Métricas (`validation/dome/metrics.py`, comunes a los tres modelos)

1. **Trayectoria media:** Y(x) = ∫ y·s dy dz / ∫ s dy dz (Legg ec. 18), promediada en el tiempo, junto con la profundidad del centroide del trazador.
2. **Transporte de la pluma:** T(x) = ∫_A U dy dz con A = {s > 0.01} (Legg ec. 11), a lo largo del talud (hacia el oeste), promediado en el tiempo.
3. **Coeficiente de entrainment:** α_E(x) = (dT/dx)/(L·Ū) (Legg ec. 12–13), con L la longitud de la interfaz s = 0.01 en el plano (y, z).
4. **Flotabilidad final:** la del agua producto, la flotabilidad media ponderada por s en la región de la pluma desprendida (como Legg Fig. 15).
5. **Mezcla:** RPE (con hipsometría exacta, como en Ilicak) y su tasa.
6. **Rangos:** de T y s, y la excursión fuera del rango inicial.
7. **Robustez:**
   - truncaciones CFL y `n_limited`;
   - velocidad máxima en caras delgadas frente a gruesas (umbral de 1 cm), con lo que se ve si aparecen los pendientes 4b y 4d;
   - si una corrida se detiene, posición y causa (post-mortem).
8. **Transporte de entrada:** total y por clase de densidad.
9. **Costo:** s por día simulado al Δt máximo estable, 1 proceso (5 hilos en pyOCEAN; MITgcm y MOM6 con 1 rank).

Las métricas se calculan con la misma función para los tres modelos sobre una malla común (la de cada resolución); los lectores convierten la salida de MOM6 (capas) y de MITgcm (z con hFac) a campos (k, j, i) con espesor.

## §8. Validación y documentación

- `validation/dome/README.md`, en el estilo de Ilicak:
  - configuración y diferencias de cada modelo, clasificadas (c);
  - tablas por resolución;
  - figuras: trazador de fondo a los 10, 20, 30 y 50 días; Y(x); T(x);
  - la lista "qué debe mejorar pyOCEAN".
- `CLAUDE.md`: filas nuevas (trazadores, esponja interior, fricción ALE, DOME) y conteo de pruebas.
- `docs/design/vision_roadmap.md` §7: fila DOME y estado de la etapa C.
- **Pruebas de la tubería** (`tests/test_dome_benchmark.py`):
  - el perfil de entrada reproduce el transporte de Legg y las capas de MOM6;
  - la batimetría coincide con la de MOM6;
  - las métricas sobre campos sintéticos con respuesta conocida;
  - los escritores de MOM6 y MITgcm producen los parámetros esperados.

## Puntos de decisión con Esteban

1. **Si MITgcm "Legg" no reproduce las Figs. 15–17 de Legg** (criterio 1), se para antes de comparar pyOCEAN: el caso está mal definido.
2. **Si una corrida de pyOCEAN se detiene** o sale de rango, se diagnostica con varias semillas antes de atribuirla. Lección del overflow de Ilicak.
3. **Si una diferencia de clase (a) requiere un cambio grande** (por ejemplo, los pendientes 4b/4d), se presenta como plan aparte.

## Restricciones

- **Recursos:**
  - ≤ 5 hilos, nunca los núcleos de eficiencia; una corrida pesada a la vez;
  - después de cada corrida MPI, revisar que no queden procesos huérfanos.
- **Salidas:** en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/`; ninguna salida, `log_*.txt` ni `*.npz` en el repo.
- **Fuentes de referencia:** no se modifican (MOM6, MITgcm, FMS). Las configuraciones van en directorios de corrida propios; los parámetros de DOME de MOM6 se cambian solo por `MOM_input`/`MOM_override`.
- **Instalaciones:** ninguna sin preguntar.
- **Git:** `git add` con rutas explícitas.
- **Corridas baratas:** a 10 km, minutos; a 5 km, ≲ 1 h por corrida.
