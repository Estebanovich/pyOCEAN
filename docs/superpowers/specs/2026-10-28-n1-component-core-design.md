# N1: contrato de componentes y driver con orden fijo

**Fecha:** 2026-10-28. **Estado:** aprobado por Esteban el 2026-10-28 (decisiones 1–3 como se proponen; 4: se mide después de N1, con la red bit a bit ya puesta).
**Rama propuesta:** `n1-component-core`, desde `main` (ac5fd09).
**Contexto:** `docs/design/vision_roadmap.md` §8 (Etapa N) y §9 (diseño en tres capas).
**Lecturas de referencia** (con `file:line`; en `docs/design/n1_refs/`):
- `mom6.md`: MOM6 0e9a561, `step_MOM` y RK2 partido.
- `mitgcm.md`: MITgcm 27256b2, `forward_step` y paquetes.
- `croco.md`: CROCO 2.1.3 (ab63b58), `step.F`.
- `oceananigans.md`: Oceananigans 0.117.0 (8ab4604), `HydrostaticFreeSurfaceModel`.
- `pyocean.md`: mapa de `SigmaBaroclinic`.

## Motivo

El núcleo 3-D es una sola clase, `SigmaBaroclinic`. Tiene unas 2300 líneas y 56 argumentos, y anida otras dos clases: `SigmaWetDry` y, dentro de esta, `WetDryFreeSurface`. Toda la física nueva que falta (S, TEOS-10, flujos de calor y E − P, KPP, mezcla por Ri, Smagorinsky, split-explicit) tendría que entrar en esa clase. JAX (Etapa D) necesita núcleos funcionales separados.

N1 extrae un contrato de componentes y un driver con orden fijo **sin cambiar la física**. La prueba es que todo quede bit a bit.

## Qué aprendimos de los cuatro modelos

| Tema | MOM6 | MITgcm | CROCO | Oceananigans | Decisión para pyOCEAN |
|---|---|---|---|---|---|
| Orden del paso | `step_MOM` fijo. Tres ritmos: DT, DT_TRACER_ADVECT y DT_THERM. ALE a ritmo DT_THERM | `forward_step` fijo. Modo síncrono o escalonado | 4 fases ocultas tras el contador `iif` | Un driver por esquema de tiempo y por tipo de superficie libre (casi duplicados) | **Un solo driver con fases con nombre**. El esquema de tiempo y la superficie libre son estrategias, no reordenamientos. Tres ritmos declarados, iguales a Δt en N1 |
| Cómo entra cada pieza | Struct de control con `_init`/`_end` y despachadores `if use_X` escritos a mano | Paquetes con puntos de entrada fijos (`APPLY_FORCING_*`, `CALC_3D_DIFFUSIVITY`) y `#ifdef` | cpp | Despacho múltiple; el componente ausente es `nothing` y su término vale 0 | **Componentes con protocolo** y un registro recorrido en orden determinista. El componente ausente devuelve cero o no se llama |
| Tendencias | El driver suma `(CAu + PFu) + diffu` en orden fijo; los diagnósticos apuntan a esos arreglos | Las de los paquetes entran por un único punto de fuentes por variable | Se escriben dentro de los núcleos, bajo `#if` | Cada término es una función; los presupuestos usan `KernelFunctionOperation` sobre las mismas funciones | **Cada término es una función pura que devuelve su arreglo**; el driver suma en el orden actual. En N4 los presupuestos llaman a las mismas funciones |
| Cierres turbulentos | `vertvisc_type` es el único canal del diabático a la dinámica | Solo producen coeficientes, reunidos en `CALC_3D_DIFFUSIVITY`/`CALC_VISCOSITY` | KPP antes del lado derecho; GLS después del corrector | `build_closure_fields`, `compute_closure_fields!`; declaran los trazadores que necesitan | **Closure** produce ν y κ, y declara su estado y los trazadores que requiere |
| Configuración y chequeos | `get_param` con unidades, valor por defecto y documentación; chequeos FATAL dispersos | `*_CHECK` + `CONFIG_CHECK`: junta todos los errores y para una vez | `check_kwds`/`check_switches2`: junta errores | Especificación → validar → materializar; mensajes que nombran el valor y la corrección | **Especificación inmutable → `check()` que junta todos los errores → `build()`**. Mensajes al estilo de Oceananigans |
| Estado | Struct de control privado; estado escondido en las structs de los componentes | Bloques common; `gU` se reutiliza para tres cosas | Common con índices de tiempo que rotan | Estado z\* dentro de la malla | **Todo lo que pasa de un paso al siguiente vive en el estado**, declarado por su componente. Nada escondido en `self` |
| Restart | Cada componente registra sus campos; checksums | Cada paquete escribe su sección; lectura por nombre | `EXACT_RESTART` guarda dos niveles | Estado prognóstico recursivo, uno por componente | **Secciones por componente**, compatibles con los restart actuales |
| Depuración | Checksums por etapa "Pre-X/Post-X"; rotación de índices | — | — | `NaNChecker` | **Checksums por fase** con `debug=True`, para localizar cualquier diferencia de bits durante N1 |

Lo que **no** copiamos:
- La configuración al compilar (MITgcm, CROCO).
- Las structs globales mutables.
- Los despachadores `if` escritos a mano.
- El despacho múltiple sobre componente × malla × esquema (no existe en Python).
- Las funciones por punto fusionadas al compilar (en Python serían lentísimas).
- Calcular la tendencia del momento al final del paso anterior (Oceananigans), porque complica el restart.

## Alcance

**Dentro de N1:**
1. El contrato de componentes (§2) y el registro de variables mínimo que lo sostiene (§3).
2. La extracción a funciones puras de la lógica que hoy vive en la clase. La lista sale del mapa (`pyocean.md` §5):
   - PGF jacobiano y de dos términos;
   - Coriolis y la composición de la tendencia del momento;
   - los drivers de advección de trazadores (sin `self._T_in`);
   - la difusión;
   - Orlanski, las esponjas y la evaluación del exterior;
   - el arrastre bbl;
   - el bloque de transportes y ω en σ;
   - la secuencia ALE.
3. El driver con fases fijas (§4) para los dos caminos (σ y ALE) y todos los esquemas actuales.
4. `SigmaBaroclinic` queda como fachada con la misma firma. Arma los componentes y delega en el driver.
5. Pre-flight `check()` (§5).
6. Restart por secciones, compatible con los archivos actuales (§6).
7. Checksums por fase en modo depuración.

**Fuera de N1** (cada uno, su propio plan):
- S, TEOS-10 y flujos de superficie: N2.
- KPP: N3.
- `Simulation`, configuración tipada y presupuestos: N4.
- El constructor de dominio: N5.
- El paquete instalable: N6.
- El split-explicit.
- Retirar el núcleo z de la Fase 1.
- Los modelos 2-D de marea (`ImplicitFreeSurface`, `WetDryFreeSurface` como modelo 2DH) siguen igual.
- **Ningún cambio de física**, aunque encontremos errores: se anotan en §8 y se corrigen después, con la red bit a bit ya en su lugar.

## Criterios de éxito

1. **Bit a bit.** `np.array_equal` en todos los campos del estado después de N pasos, en la matriz dorada (§7), y los 1388 tests pasan sin tocar tolerancias.
2. **Benchmarks.** Corridas cortas de humo, idénticas bit a bit a `main`:
   - corte BayIW z\* e híbrida, 1 h;
   - lock exchange de Ilicak z\*/σ/híbrida, 30 min;
   - DOME a 10 km, 1 día;
   - onda interna, 1 h;
   - seamount σ, 1 día.
3. **Costo.** ≤ 2 % por paso en el corte BayIW z\* e híbrida (5 hilos, máquina libre) y en el overflow de Ilicak.
4. **Restart.** Un restart escrito por `main` antes de N1 se lee y continúa bit a bit (archivo guardado en `tests/data/`).
5. **Nada privado desde fuera.** Los runners de `validation/` dejan de llamar métodos privados: usan accesores públicos nuevos (`face_geometry()`, `mean_face_thickness()`, `baroclinic_pgf()`). Los nombres privados y los del módulo que hoy importan los tests quedan como alias con aviso de obsolescencia.

## §1. Las tres capas

```
Capa 3  Usuario      SigmaBaroclinic (fachada, misma firma)   →  en N4: Model / Simulation
Capa 2  Componentes  especificación (dataclass frozen, sin arreglos)
                     → check(ctx) → build(grid, ctx) → componente en marcha
Capa 1  Núcleos      funciones puras sobre arreglos (NumPy/Numba), no modifican la entrada
                     salvo donde se declara (ver §4, riesgos)
```

- **Paquetes nuevos:**
  - `core/model/`: `driver.py`, `contract.py`, `registry.py`, `state.py`.
  - `core/components/`: un módulo por componente.
- Los núcleos que ya son puros se quedan donde están (`core/vertical/*`, `vector_invariant`, `open_boundary`, `bottom_friction`, `physics/mixing/*`).

## §2. Contrato de componente

Es un `typing.Protocol`. Cada componente declara:

| Elemento | Contenido | Inspirado en |
|---|---|---|
| `name`, `role` | Rol: `eos`, `vertical_coordinate`, `pgf`, `momentum`, `time_stepper`, `free_surface`, `closure`, `bottom_drag`, `layer_transport`, `tracer_advection`, `tracer_diffusion`, `open_boundary`, `interior_sponge`, `surface_fluxes` o `diagnostics` | roles de MITgcm/MOM6 |
| `params` | Campos del dataclass con unidades, valor por defecto, cita y benchmark que lo validó (metadatos del campo) | `get_param` de MOM6 |
| `reads`, `writes` | Variables del registro con su **nivel de tiempo** (n, n+1, †) y su fase | lección de MITgcm sobre niveles no documentados |
| `state_fields()` | Lo que el componente pasa de un paso al siguiente: estado de GLS, exterior, LU desfasado | MOM6 `register_restart_field`, `prognostic_state` de Oceananigans |
| `requires()` | Otros roles o variables que necesita (por ejemplo, el FCT necesita ALE) | `required_tracers` de Oceananigans |
| `check(ctx) -> list[Issue]` | Errores y avisos; no lanza excepciones | `*_CHECK` de MITgcm |
| `build(grid, ctx)` | Arreglos fijos y cachés | `materialize_*` de Oceananigans |
| Métodos por fase | Por ejemplo, `tendency(state, grid) -> dict[str, ndarray]` o `apply(state, ...)` | — |

**Componentes de N1.** Son las costuras que propone el mapa (`pyocean.md` §7):
- EOS: lineal en T, como hoy.
- VerticalCoordinate: σ o regla ALE con regrid y remapeo.
- PGF: jacobiano, de dos términos o de volumen finito.
- Momentum: forma de flujo o vector-invariante, con la viscosidad horizontal.
- TimeStepper: AB3 para el momento; SSP-RK3 o predictor–corrector para los trazadores.
- FreeSurface+VerticalViscosity: envuelve `SigmaWetDry.step` tal como está.
- Closure: constante, parabólica o GLS.
- BottomDrag: log-law o cuadrático en hbbl.
- LayerTransport: transportes en σ, o flujo de grosor PPM/upwind con el limitador de salida.
- TracerAdvection: WENO5 con o sin FCT.
- TracerDiffusion.
- OpenBoundary: Flather, Orlanski, esponja de borde, exterior y caras 'clamped'.
- InteriorSponge.
- SurfaceFluxes: hoy solo τ.
- Diagnostics: `mix_num`, `n_trunc`, `n_limited`.

**FreeSurface+VerticalViscosity se envuelve sin partir.** `SigmaWetDry.column_step` mezcla el cierre, el arrastre, el viento y la corrección δ, y su orden de operaciones es delicado para el bit a bit. Partirlo es una tarea de N2/N3: se hará cuando el cierre tenga que dar también κ para los trazadores.

## §3. Estado y registro de variables

- **`SigmaState` pasa a contener todo lo que cruza pasos.** Hoy está escondido en `self` (`pyocean.md` §2):
  - el estado de GLS;
  - el exterior `T_ext`/`h_ext`/`tr_ext`;
  - `rho_ref`, después de `initial_state`;
  - la geometría `fs.last_*`;
  - el precondicionador LU desfasado.

  El LU se clasifica como **caché que afecta los bits**: va al restart, igual que hoy.
- **Clasificación de cada campo:**

  | Clase | Ejemplos |
  |---|---|
  | prognóstico | η, u, v, T, h, tr |
  | historia | `g_hist`, Qx/Qy/Fz/ω del paso anterior |
  | estado de componente | GLS, LU, exterior |
  | diagnóstico | `mix_num`, `n_trunc`, `n_limited` |
  | caché reconstruible | `_tt_cache`, `_zc_cache`, `_lap` |

  Solo las cachés reconstruibles pueden quedar en el componente.
- **Las variables de paso que hoy viajan por `self`** (`_ext`, `_T_in`, `_tr_in`) pasan a ser argumentos explícitos entre fases.
- **El registro de variables mínimo** (`core/model/registry.py`) guarda para cada variable: nombre, dims, ubicación en la malla C, unidades, nombre CF y clase. Lo usan el restart y `check()`; la salida CF lo adopta en N4. Es el "registro único" de CLAUDE.md, iniciado aquí con los campos de `SigmaState`.

## §4. Driver: fases fijas

Un solo `Driver.step(state, forcing) -> state`. Las fases reproducen **exactamente el orden actual** (`pyocean.md` §3). Cada fase llama a los componentes de su rol en un orden fijo y determinista.

| # | Fase | σ | ALE |
|---|---|---|---|
| 1 | `prepare` | exterior (si hay proveedor) | geometría de caras, exterior |
| 2 | `explicit_momentum` | PGF → += Coriolis o vector-invariante → += advección → += laplaciano → −= advección vertical (mismo orden y mismos paréntesis que `SB:1579-1614`) | igual |
| 3 | `time_combine` | AB3 (AB1/AB2 al arrancar), τ/ρ₀ sin extrapolar | igual |
| 4 | `implicit_and_free_surface` | `SigmaWetDry.step`: viscosidad implícita, log-law, viento, superficie libre θ, Flather, GLS | grosor viscoso y arrastre bbl → `core.step` (CG + LU desfasado, conservación exacta, δ) |
| 5 | `boundary_velocity` | esponja y Orlanski | esponja, Orlanski, capas delgadas de fondo, mezcla opcional |
| 6 | `layer_transport` | transportes desde la geometría de la superficie libre (θ, θ_bc), Fz/ω | `transport_velocity`, flujo de grosor, caras abiertas, `limit_outflow`, h† |
| 7 | `tracers` | SSP-RK3 WENO5 de [T, T²], `mix_num`, kh, kv, esponja de trazadores | FCT de [T, (T²), tr], kh, kv, esponja de borde sobre h† |
| 8 | `vertical_coordinate` | — | regrid y remapeo |
| 9 | `interior_sponge` | — | esponja interior |
| 10 | `finalize` | — | remapeo de velocidad en fracción, tope de CFL, contadores |

- **Ritmos.** El driver declara `dt_dyn`, `dt_tracer` y `dt_therm` (MOM6). En N1 se exige que los tres sean iguales; `check()` da error si no. Separarlos después (ALE a ritmo DT_THERM) es una mejora de costo con su propio benchmark.
- **Términos con nombre.** La fase 2 obtiene `{"pgf": …, "coriolis": …, "advection": …, "hviscosity": …, "vadvection": …}` y los suma en el orden de hoy. Con `keep_terms=True` el driver los guarda, sin cambiar los bits, para los presupuestos de N4.
- **Checksums por fase.** Con `debug=True` se calcula un hash del estado al final de cada fase. Contra `main` instrumentado, eso localiza la primera fase que diverge.
- **Riesgos para el bit a bit** (`pyocean.md` §7), con su mitigación:
  - **Orden de suma.** No se unifica `np.tensordot` con `(ds*u).sum(0)`; cada expresión se mueve tal cual.
  - **Escrituras in situ** (el borde escribe en `u_new` y `Qx`). Se conservan, y el contrato las declara (`writes_inplace`).
  - **Cachés por identidad de arreglo.** No se copian `h` ni `T` entre fases; el driver pasa las mismas referencias.
  - **Despacho NumPy/Numba con condiciones exactas** (`_diffuse_h`, `ReferenceProfile` con ≥ 1000 puntos, `_rhs`). Se mueven sin cambiar las condiciones.
  - **`_THREADS`.** Un solo lugar; los módulos nuevos lo importan.

## §5. Pre-flight

- **Cómo funciona:**
  - `check()` recorre todos los componentes y la tabla de compatibilidad.
  - Junta todos los `Issue` (error o aviso), los muestra juntos y lanza **una** excepción si hay errores. Así lo hacen `CONFIG_CHECK` de MITgcm y `check_kwds` de CROCO.
  - Cada mensaje nombra el valor, las opciones válidas y la corrección (Oceananigans).
- **Los chequeos que hoy existen** (`SB:1050-1220`) pasan aquí tal cual: mismo resultado, mejor mensaje.
- **Los argumentos que hoy se ignoran en silencio** (`pyocean.md` §1) dan **aviso** en N1, no error, para no romper scripts existentes:
  - en σ: regrid, regrid_params, remap_scheme, h_thin_bed, init_iterations, max_truncations, thin_layer_cutoff, mix_diagnostic, thickness_flux y thin_layer_coupling='mom6';
  - `ke_scheme` con la forma de flujo;
  - `sponge_T`/`sponge_zw` sin `sponge_rate`.
- **`adv` sin validar pasa a error.** Un valor inválido hoy ya falla, solo que más tarde.
- **`ke_scheme="gudonov"`** se acepta `"godunov"` como nombre correcto, y `"gudonov"` queda como alias con aviso (principio 5 de §9).

## §6. Restart

- **El archivo se organiza por secciones de componente.** Cada componente escribe sus `state_fields()`, y la huella de configuración se arma con los `params` de cada componente. Hoy se leen los privados `_sp_T`/`_sp_zw`.
- **Lectura compatible:** si el archivo tiene el formato actual, se mapea campo por campo.
- Prueba con un restart de `main` guardado en `tests/data/n1_restart_main.nc`, de 5 × 6 × 4 celdas y menos de 100 kB.

## §7. Verificación

1. **Tarea 0, antes de mover código:**
   - Un script genera la **matriz dorada** desde `main`: estados completos después de 1, 2, 3 y 10 pasos en mallas pequeñas. Los tres primeros cubren el arranque de AB1 a AB3.
   - Cubre todas las combinaciones que el código distingue:
     - coordenada: σ, z\*, híbrida;
     - esquema de trazadores: SSP-RK3 o predictor–corrector;
     - momento: forma de flujo o vector-invariante;
     - KE: Arakawa o Godunov;
     - flujo de grosor: PPM o upwind;
     - PGF: jacobiano, dos términos o volumen finito;
     - cierre: constante, parabólico o GLS (σ);
     - arrastre: log-law o cuadrático;
     - bordes: cerrados, Flather + Orlanski + esponja, 'clamped' con proveedor;
     - esponja interior;
     - trazadores pasivos;
     - `mix_diagnostic`;
     - acoplamiento de capas delgadas.
   - Son unas 30 configuraciones, en `tests/data/n1_golden/`, con un test `test_n1_golden.py` que compara con `np.array_equal`.
   - Se guardan también los checksums por fase de `main`, con una instrumentación temporal que no entra a `main`.
2. **Durante la extracción.** Cada componente movido se comprueba con la matriz dorada antes de seguir; un commit por componente.
3. **Al final:**
   - la suite completa;
   - las corridas de humo del criterio 2;
   - el costo del criterio 3;
   - el restart del criterio 4.

## §8. Hallazgos que no entran en N1 (tareas aparte)

1. **La historia de AB3 no se remapea después del ALE** (`SB:2254` → `_step_ale` → `:2241`). Las tendencias de los pasos n−1 y n−2 se aplican por índice de capa después de que el regrid movió las capas. MOM6 remapea `u_av`, `CAu_pred` y `diffu` (`mom6.md` §6). Es candidato (a).
   - En z\* las capas se mueven poco.
   - En la híbrida puede contribuir al ruido en las capas evanescentes y a la fase de BayIW 3-D.
   - Hay que medir el efecto antes de decidir.
2. **El cierre solo da ν al momento.** Los trazadores usan siempre `kv` constante, y GLS no tiene término de flotabilidad (b). Se resuelve en N2/N3, con Closure → (ν, κ).
3. **ALE en cada paso** frente a MOM6, que lo hace a ritmo DT_THERM: oportunidad de costo.
4. **`fv_pgf` está atado a la EOS lineal.** N2 necesita una integral de densidad que acepte cualquier EOS (`int_density_dz` de MOM6).
5. **KPP y `forcing/surface.py` están escritos para el `ModelState` de la Fase 1** (k = 0 arriba). N3 los reescribe por columna.
6. **El momento es siempre AB3.** `"ssprk3"`/`"pc"` solo aplican a los trazadores. Se documenta en el contrato del TimeStepper; la API de N4 debería separar los dos esquemas en dos argumentos.

## §9. Orden de trabajo (para el plan)

0. Matriz dorada y checksums por fase desde `main`.
1. `core/model/contract.py`, `registry.py` y `SigmaState` extendido (sin cambiar los valores).
2. Extraer los núcleos uno por uno, en el orden de las fases: PGF y momento → TimeStepper → envoltura de FreeSurface → borde de velocidad → transporte de capas → trazadores → coordenada vertical y esponja interior → finalize. Después de cada uno se corre la matriz dorada.
3. `Driver` con las fases de §4; `SigmaBaroclinic` como fachada.
4. Pre-flight `check()`.
5. Restart por secciones y lectura del formato anterior.
6. Accesores públicos, alias de obsolescencia y los dos monkeypatch de los tests apuntando al nuevo lugar.
7. Corridas de humo de los benchmarks y medición de costo.
8. Documentación:
   - el manual (arquitectura y paso de tiempo);
   - CLAUDE.md;
   - el §9 del roadmap;
   - el docstring de cada componente con su contrato.

## Puntos de decisión con Esteban

1. **Envolver `SigmaWetDry.step` sin partir en N1** (propuesto) o partirlo ya en viscosidad, arrastre, superficie libre y GLS.
2. **Argumentos ignorados hoy:** aviso en N1 (propuesto) o error.
3. **Copias de las lecturas de referencia** en `docs/design/n1_refs/` (hechas; ~17 000 palabras): ¿se quedan en el repo?
4. **El hallazgo §8.1 (historia de AB3 sin remapear):** ¿se mide antes de N1, en un scratch rápido, o se agenda para después?

## Restricciones

- Ningún cambio de física ni de tolerancias en los tests.
- Commits con rutas explícitas; nunca `git add validation/`.
- ≤ 5 hilos y solo P-cores. Las corridas de humo se hacen una a la vez.
- Python, PEP 8, type hints y docstrings con la física y sus citas (CLAUDE.md).
