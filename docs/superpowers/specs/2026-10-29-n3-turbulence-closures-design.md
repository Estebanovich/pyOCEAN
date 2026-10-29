# N3: cierres turbulentos en σ y ALE (GLS k-ε estratificada y KPP) y flujo de calor de superficie

**Fecha:** 2026-10-29. **Estado:** borrador para revisión de Esteban (revisión de constructor hecha; las dos preguntas abiertas las contestó Esteban: ley de pared en ALE y fronteras periódicas, ambas dentro).
**Contexto:** etapa N de `docs/design/vision_roadmap.md` §8. Viene después de N2a (salinidad, agua dulce real, balance exacto de varianza, TEF, GLS estratificada en σ, benchmark de Lorenz; en `main` desde `b256983`). N2b (TEOS-10 y flujos bulk) va después de este trabajo.
**Lecturas de referencia** (con `archivo:línea`) en `docs/design/n2c_refs/`:
- `tke_in_ale.md`: HYCOM, MOM6, CROCO y Oceananigans;
- `croco_column_cases.md`: los siete casos de columna de CROCO y su KPP;
- `kpp_mom6_cvmix.md`: la KPP de MOM6/CVMix, los `CVmix_SCM_tests` y lo que le falta a `physics/mixing/kpp.py`;
- `closure_algorithms_literature.md`: lo que dice la literatura local sobre cada decisión numérica (variable de longitud, positividad, producción P y B, ubicación de k y ε, KPP en capas no uniformes, onda corta).

## Por qué

En el modo ALE pyOCEAN no tiene cierre turbulento. Hoy solo admite ν constante o ninguno (`core/model/preflight.py:130`), y la viscosidad constante no tiene ajuste convectivo. Las consecuencias medidas son dos:

- **Lorenz** (`validation/lorenz_estuary/README.md`, "Resultados"): ALE-σ mezcla ~3 veces más que GETM en A, casi no forma el tapón salino en B y C, da ~6 veces el flujo de intercambio en D y borra la capa de lluvia en C. Todo esto es clase (c), con la (b) "sin cierre en ALE" detrás.
- **DOME** (`validation/dome/README.md`): el híbrido da b_f −16 × 10⁻³ contra el −12 de Legg, porque falta la mezcla por Richardson. Es clase (b).

ALE (z*, híbrido) es la coordenada con la que pyOCEAN se acerca a HYCOM y a las ondas internas de la tesis de Esteban. Un cierre que solo existe en σ deja fuera el camino principal.

KPP entra en el mismo trabajo por decisión de Esteban. Es el segundo cierre del roadmap (etapa N) y el que usan HYCOM-IOA y MOM6. Existe en `physics/mixing/kpp.py` desde la Phase 1, pero en z-levels y sin conexión al núcleo 3-D.

Para probar los cierres con convección y calentamiento, el modelo necesita un flujo de flotabilidad de superficie. Hoy `step()` solo recibe viento y `fw`. Por decisión de Esteban se agrega aquí un **flujo de calor prescrito simple**; las fórmulas bulk y TEOS-10 se quedan en N2b.

## Criterios de éxito (decididos con Esteban)

1. **Columnas contra CROCO (GLS).**
   - Kato–Phillips: la profundidad de la capa de mezcla queda a ±5 % de CROCO y de la fórmula de Price (1979) [verificar en Umlauf et al. 2005 o Price 1979].
   - Willis–Deardorff, Ekman de fondo, doble Ekman, capa de fondo sin rotación y su versión oscilante: los perfiles de T, u, k y ε siguen a CROCO con la misma resolución.
2. **Columnas contra MOM6/CVMix (KPP).** En `wind_only`, `cooling_only` y `mech_only`, la profundidad de la capa límite y la de mezcla quedan a ±5 % de MOM6. `skin_warming_wind` se reporta.
3. **σ contra ALE.**
   - En una columna con z* y capas fijas, cada cierre en ALE da lo mismo que en σ, a redondeo o casi (la tolerancia se fija y justifica en el plan).
   - Con capas que se mueven, el remapeo conserva la integral de k en la columna a redondeo y nunca da k < k_min ni ε < ε_min.
4. **Lorenz ALE-σ + GLS.**
   - Se acerca a GETM y a nuestro σ+GLS en el S máximo, Me, Q_in y s_in.
   - Cada diferencia restante se clasifica (a)–(d).
   - El balance de varianza sigue cerrando a ≤ 5e-10.
5. **DOME híbrido + GLS.** b_f se mueve de −16 hacia −12 (Legg), con T exactamente en rango y sin truncamientos CFL nuevos.
6. **Nada se rompe.**
   - Con `closure="constant"` o `"none"`, y con la GLS de σ con su valor por defecto, la matriz dorada de N1 y el smoke de 9 casos quedan bit a bit.
   - El costo de esos casos no sube más de 1 %.
   - Los tests actuales pasan, salvo los que fijen un error de signo de `kpp.py`; esos se corrigen con su justificación en el plan.

**Anti-objetivos.** Cualquiera de estos cuenta como fracaso aunque el código funcione:
- Calibrar constantes para acercarse a un caso. Se usan las de CROCO (GLS, Canuto A) y las de CVMix (KPP) tal cual, citadas.
- Que KPP traiga activa por defecto una difusividad de fondo. La tesis de Esteban mostró que daña los espectros de ondas internas; por defecto es 0 y es configurable.
- Que la mezcla numérica del remapeo de k y ε domine sobre la física. Se mide con el balance de varianza o con un diagnóstico equivalente para k.
- Que la GLS en ALE solo sea estable con un Δt mucho menor que en σ, a igual resolución.

## Alcance

**Dentro:**

1. **Flujo de calor de superficie prescrito.**
   - `step(st, tau_x, tau_y, fw=None, heat=None)`, donde `heat` es un dict con:
     - `Q_ns`: flujo no solar [W m⁻², positivo hacia el océano] por celda;
     - opcionalmente `Q_sw`: onda corta [W m⁻²], que penetra con un perfil de Jerlov (Paulson & Simpson 1977, dos bandas; tipo I por defecto; CROCO `lmd_swfrac`, ver `croco_column_cases.md`).
   - Entra como flujo de T por la interfaz superior, y la onda corta como fuente en las capas según la fracción que llega a cada interfaz. La parte que llega al fondo se deposita en la capa del fondo, como hacen MOM6 y CROCO (el constructor confirma cuál regla usa cada uno y cita `archivo:línea`).
   - En σ y en ALE. El balance de varianza (N2a) gana un término de superficie para T, análogo al de E−P.
   - Restart: no hay estado nuevo; el flujo es un argumento de `step()`.
   - Sin `heat` (valor por defecto) todo queda bit a bit.
2. **El cierre como componente propio.**
   - Hoy la GLS vive dentro del paso de superficie libre de σ (`core/components/free_surface.py:6-9`, `sync_closure_in`/`closure_out`). Se extrae a un componente `Closure` con una interfaz común para `'constant'`, `'none'`, `'gls'` y `'kpp'`.
   - **Entrada:** u, v, la densidad de las capas, el espesor de las capas (σ o ALE), u* de superficie y de fondo, y los flujos de flotabilidad de superficie.
   - **Salida:** ν en interfaces, κ_T y κ_S en interfaces y, para KPP, el término no local.
   - La difusión vertical implícita existente usa esas salidas.
   - El orden de fases del driver de N1 se respeta; si hay que mover la fase del cierre se justifica, y el bit a bit (criterio 6) lo controla.
   - Esto también cumple lo pendiente de N1 y N2: separar viscosidad, arrastre, superficie libre y cierre.
3. **GLS k-ε estratificada en ALE.**
   - El mismo código que en σ (`physics/mixing/gls.py`, `gls_stratified=True`), sin una segunda implementación.
   - **Cómo viajan k y ε en ALE** (decisión 3 abajo; antecedentes en `tke_in_ale.md` y `closure_algorithms_literature.md`): sin advección horizontal (igual que en σ); tras cada regrid se remapean en la vertical como contenido h·k y h·ε, con un esquema que no da negativos (como HYCOM remapea q² y q²l en `hybgen.F90`, y Oceananigans conserva e·σ en z*). Entran al restart.
     - La GLS resuelve en interfaces, como CROCO. La conversión interfaz → capa antes del remapeo y capa → interfaz después debe conservar la integral de k en la columna. El constructor elige el esquema y lo justifica.
     - Pisos: k ≥ k_min y ε ≥ ε_min después de cada operador, los de CROCO (`tke_min` 1e-10, `eps_min` 1e-12), y el límite de Galperin.
   - **Capas delgadas o desaparecidas:**
     - N² y S² se calculan entre centros de capas con un piso de espesor. La referencia es MOM6, que junta las capas sin masa antes de calcular gradientes (`MOM_kappa_shear.F90`), y HYCOM, con su piso `dp` de 1 mm y `klist`.
     - Una capa sin masa no debe crear un pico de corte ni de N².
     - El constructor elige la regla, la cita y la prueba con una columna híbrida con capas casi vacías.
   - **Condición de fondo (decisión de Esteban: ley de pared en ALE).** La ley de pared (`z0`, C_d(D, z₀) de `physics/bottom_friction.py`) y la condición de fondo de la GLS para k y ε pasan a ALE, igual que en σ. Así σ y ALE se comparan limpio en las columnas con esfuerzo de fondo y en Lorenz.
     - Problema que el constructor resuelve y documenta: en z* o en el híbrido la capa del fondo puede ser muy delgada o desaparecer (por eso hoy ALE exige `z0=None`). La distancia al fondo para la ley de pared se toma sobre un espesor mínimo (por ejemplo, el de la capa o capas de fondo que suman `hbbl`, o el espesor de la primera capa con masa), con la referencia de cómo lo hace MOM6 o HYCOM, citada.
     - El arrastre cuadrático sobre `hbbl` (`bottom_drag='quadratic'`) se mantiene como opción en ALE; DOME lo sigue usando como línea base.
4. **KPP en σ y ALE** (`closure="kpp"`).
   - La referencia es CVMix tal como lo usa MOM6:
     - `MOM_CVMix_KPP.F90`, `pkg/CVMix-src/src/shared/cvmix_kpp.F90`;
     - capa límite por Richardson bulk;
     - forma σ(1−σ)² sin empate con el interior;
     - término no local 6.327 σ(1−σ)², solo con flujo desestabilizador;
     - mezcla aumentada en la interfaz de la base;
     - Vt² con los parámetros de CVMix.
     Cada elección se cita en el código.
   - Se reescribe sobre `physics/mixing/kpp.py` y se corrige su **error de signo** (`b_flux > 0` tratado como desestabilizador; ver `kpp_mom6_cvmix.md` §7–8). Es clase (a): primero un test que falle y después la corrección. El test `test_surface_heating_produces_nonlocal_warming`, que hoy fija ese error, se corrige.
   - **Lo que le falta a `kpp.py`:**
     - índice k = 0 en el fondo, como el núcleo;
     - geometría a partir del h 3-D;
     - casos límite con u* = 0;
     - ρ₀, g, α y β del modelo y no fijos en el código;
     - S;
     - flujos no solar y onda corta separados.
   - **Opcional y en 0 por defecto:**
     - difusividad de fondo;
     - mezcla interior por cizalla de Large et al. 1994 (LMD94);
     - capa límite de fondo.
     El constructor decide si la mezcla interior por Ri entra como parte de `'kpp'` (como CROCO y CVMix) y lo deja documentado. Para DOME interesa la de la GLS, no la de KPP.
5. **Fronteras periódicas en x e y** (decisión de Esteban). Opción del núcleo (`periodic=("x",)`, `("y",)` o ambas, o el nombre que elija el constructor) en σ y ALE: caras de u/v, advección, PGF, superficie libre (matriz del solver), FCT y restart. Las columnas de CROCO (5×5) y de MOM6 (2×2) son dominios doblemente periódicos; también sirven para canales. Sin la opción todo queda bit a bit. Pre-flight rechaza combinarla con fronteras abiertas en el mismo eje.
6. **Validación, en este orden:**
   1. **Columnas de CROCO** (`croco_column_cases.md`): Kato–Phillips, Willis–Deardorff, ciclo diurno, FORCED_EKBBL, FORCED_DBLEEK, FORCED_NONROTBBL y FORCED_OSCNONROTBBL.
      - Se compilan en `~/pyOCEAN_refs/croco_runs/`, como thacker.
      - Se corren en pyOCEAN σ y ALE (z* e híbrido donde tenga sentido) con la GLS, y Kato–Phillips, Willis–Deardorff y el ciclo diurno también con KPP.
      - Viven en `validation/turbulence_columns/` (o el nombre que elija el constructor), con definición del caso, runner, comparación y README.
   2. **Columnas de MOM6** (`CVmix_SCM_tests`, `kpp_mom6_cvmix.md`): `wind_only`, `cooling_only`, `mech_only` y `skin_warming_wind`, con el MOM6 compilado (`~/pyOCEAN_refs/MOM6/.testing/build/symmetric/MOM6`, igual que `validation/mom6_lock_exchange/run_mom6.sh`), con KPP en pyOCEAN σ y ALE z*.
   3. **Lorenz ALE-σ + GLS:** cuatro corridas nuevas con el runner existente (configuración nueva `ale_sigma_gls`), contra GETM, nuestro σ+GLS y ALE-σ constante.
   4. **DOME híbrido + GLS** a 10 km, con el runner existente.
   - Cada benchmark termina con la clasificación (a)–(d) y una lista de "Qué debe mejorar pyOCEAN".
   - Las salidas van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`, nunca al repo.
7. **Documentación:**
   - filas en CLAUDE.md;
   - scorecard §7 y §8 de `vision_roadmap.md`;
   - manual (paso de tiempo, cierres);
   - `reading_list.md`, con los artículos de Consensus que se usen: Umlauf et al. 2005, Warner et al. 2004, Burchard et al. 2001, Reffray et al. 2014, Li et al. 2021 y Halliwell 2003.

**Fuera:**
- TEOS-10 y fórmulas bulk (N2b);
- Langmuir y oleaje;
- Smagorinsky o viscosidad horizontal;
- ePBL de MOM6;
- MY2.5 y GISS de HYCOM;
- ajuste convectivo separado (con la GLS estratificada es casi redundante, según Gualtieri et al. 2026);
- ajustar constantes;
- extender las corridas B, C y D de Lorenz (se hará después, con la tabla definitiva).

## Restricciones

- **Python solamente.** Numba en los bucles calientes, con un oráculo NumPy en los tests, como el resto del núcleo.
- **Referencias en el código:** cada esquema cita su artículo, y cuando lo toma de un modelo de producción, `archivo:línea` del código de referencia.
- **Corridas largas:**
  - una a la vez, ≤ 5 hilos y solo en los P-cores;
  - siempre con `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`;
  - si una corrida pasa de ~24 h de reloj, se consulta a Esteban.
- **Commits con rutas explícitas.** Nada de `logs/`, `.npz` ni salidas; push solo si Esteban lo pide.
- **Restart bit a bit con los cierres nuevos.** El estado de la GLS en ALE (k, ε y los coeficientes que pasan de un paso al siguiente) y cualquier estado de KPP que lo necesite entran al restart, etiquetados por componente como en N1. Un restart de otra configuración de cierre se rechaza con un error claro.

## Decisiones de diseño tomadas por Claude (Esteban las revisa al leer)

Salen de la revisión de constructor y de las lecturas de referencia y literatura.

1. **k y ε, no otra variable de longitud.** Con los modelos calibrados el resultado es casi igual (Umlauf & Burchard 2003; Klingbeil et al. 2018), k-ε no necesita función de pared y es lo que ya está portado de CROCO. Constantes de CROCO, Canuto A, tal cual.
2. **k y ε viven en interfaces** (como CROCO); N² y S² entre centros de capa. Las capas más delgadas que un espesor mínimo se excluyen de la columna del cierre o se juntan con la vecina (MOM6 `eliminate_massless`), sin crear picos de N² ni de corte.
3. **Sin advección horizontal de k y ε, ni en σ ni en ALE**, igual que hoy en σ (`physics/mixing/gls.py`: no se portó `GLS_HADV_UP1`). En ALE solo se remapean en la vertical tras el regrid, como contenido h·k y h·ε con un esquema que no da negativos, seguido de pisos y del límite de Galperin. Así σ+GLS y ALE+GLS tienen la misma física. Que el remapeo sea así es juicio de diseño (no hay referencia en la literatura local); se prueba con Kato–Phillips con y sin regrid.
4. **Positividad:** términos de pérdida linealizados dentro del solver implícito (Patankar; Klingbeil et al. 2018), luego pisos y límite de Galperin. Nada de recortar a mano.
5. **Producción P y B que conserva energía** (Burchard 2002): la energía que pierde el flujo medio por la viscosidad y la mezcla de densidad entra exacta a k. El constructor lee el artículo (por descargar) y cita la forma. Encaja con el balance exacto de varianza de N2a.
6. **El flujo de calor no entra directo a la GLS:** actúa solo a través de N², como en CROCO `gls_mixing.F`. Para KPP sí entra como flujo de flotabilidad de superficie.
7. **KPP en un módulo nuevo** con la geometría del núcleo (por ejemplo `physics/mixing/kpp_cvmix.py`). `physics/mixing/kpp.py` (Phase 1, z-levels) se queda con su API y sus tests, solo con la corrección del signo y la del test que lo fijaba. La KPP nueva sigue a CVMix como lo usa MOM6, sin empate con el interior; si hacen falta iteraciones del paso semi-implícito (Large et al. 1994, Apéndice D; HYCOM usa 2), se citan.
8. **Referencia KPP en las columnas de CROCO:** se compila también CROCO con KPP (`LMD_SKPP`) para Kato–Phillips, Willis–Deardorff y el ciclo diurno. Las diferencias por constantes distintas (CROCO 2005 vs CVMix) se clasifican (c); el criterio de aprobado de KPP es contra MOM6/CVMix y contra la teoría (Kato–Phillips: D_m = 1.05 u* N₀^(−1/2) t^(1/2), Warner et al. 2005 ec. 60).
9. **Columnas de MOM6 con su configuración exacta:** 400 capas de 1 m, 2×2 periódico, Δt 1200 s; en pyOCEAN ALE z* y σ con los mismos niveles.
10. **DOME + GLS:** solo cambia el cierre respecto a la línea base de DOME (arrastre cuadrático sobre `hbbl` igual); el piso de κ_t es el de la GLS (kv = 0 → 1e-6 de CROCO). Si además se corre con el `kv` viejo como piso, se reporta aparte, sin elegir el que se acerque más.

## Lo que decide el constructor (y lo documenta)

- La conversión interfaz ↔ capa para k y ε en el remapeo.
- La regla exacta de capas delgadas para N² y S² y la distancia al fondo de la ley de pared en ALE.
- La tolerancia exacta σ contra ALE (criterio 3).
- Si la mezcla interior por Ri entra en `'kpp'`.
- La ubicación de los casos de columna dentro de `validation/`.
- La regla de onda corta que llega al fondo.
