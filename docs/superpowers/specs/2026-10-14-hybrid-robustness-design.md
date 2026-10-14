# Robustez de la híbrida: capas casi vacías, grosor positivo y T acotada

**Fecha:** 2026-10-14. **Estado:** diseño aprobado por Esteban en la conversación por secciones; falta revisar el texto escrito.
**Rama:** `hybrid-robustness`, desde `main` (abe9be9).

## Motivo

El benchmark de Ilicak (`validation/ilicak_benchmark/README.md`, puntos 3, 4 y 9) dejó tres defectos de clase (a) en la híbrida de pyOCEAN:

1. **Se detiene cuando se vacía una capa** ("layer emptied in the Lagrangian step"):
   - en el overflow con ν ≤ 100;
   - en la onda interna con ν = 0.01 si θ_bc = ½.

   Por eso la híbrida conserva θ_bc = θ = 0.6 y todavía amortigua la onda interna (0.45 frente a 0.65–0.72).
2. **T fuera de rango en capas delgadas:** 3.2–31 °C con 5–30 en el lock exchange, y hasta 33 °C en el overflow.
3. **Velocidades de 47 m/s** en el overflow con ν = 1000.

Además, WENO5-Z crea extremos nuevos de T en z\* (hasta 4.7 °C en el 0.25 % del volumen del overflow; punto 2 de la lista).

## Diagnóstico previo (2026-10-14, scripts de scratch)

- **La raíz son las velocidades de las capas casi vacías.**
  - En el overflow híbrido con ν = 100 y θ_bc = 0.6, las caras con ambos lados de al menos 1 m llegan a ~10 m/s desde las 3 h. Es físico: la caída libre por el talud da ~8 m/s en todos los modelos.
  - Las caras con un lado de menos de 1 cm crecen de 2 a 24 m/s entre las 2 y las 5.5 h. A las 6 h todo explota (45–50 m/s).
  - T sale del rango 10–20 °C desde las 3 h (6.5 °C a las 4.5 h), antes de la explosión.
  - En la onda interna con θ_bc = ½, una capa de 1 mm tiene Courant 0.45 y 0.52 en sus dos caras (~8 m/s en una onda de 0.4 m/s).
- **El vaciado lo produce la corrección barotrópica, no PPM.**
  - Onda interna: el flujo PPM sin corregir saca el 99.4 % del contenido de la capa; la corrección (el du uniforme por cara más el reparto del residuo) añade otro 1.2 %.
  - Overflow con ν = 100: 90.6 % más 10 %.
- **Lo que hacen las referencias** (lectura de código, 2026-10-14):
  - **MOM6:**
    - acota la corrección du por capa con el CFL (`MOM_continuity_PPM.F90:783-874`, la variante "agresiva" en :848-851);
    - acepta un desajuste entre los transportes barotrópico y baroclínico antes que vaciar una capa (:1542-1544, `eta_cor` en `MOM_barotropic.F90:5432-5485`);
    - acopla las capas delgadas con la viscosidad vertical (`MOM_vert_friction.F90:1457-1532, ~2236`);
    - limita el flujo de trazador a la masa disponible, con una reconstrucción monótona (`MOM_tracer_advect.F90:573-600, 741-749`).
  - **HYCOM:**
    - usa FCT de Zalesak para el grosor (`cnuity.F90:412-468`) y para los trazadores (`mod_tsadvc.F90:786-791, 877-905, 972-976`);
    - mezcla la velocidad de las capas de menos de 0.5 m con la de la capa de arriba (`momtum.F90:3055-3062`).
  - **Ninguna de las dos completa este overflow con ν ≤ 100:** MOM6 ρ trunca velocidades y HYCOM se detiene por `neg. dp`.

## Alcance

Es el modo ALE de `SigmaBaroclinic` (`core/vertical/ale.py`, `core/dynamics/sigma_baroclinic.py`, el núcleo de la viscosidad vertical y `core/restart.py`).
- Las piezas 1 y 3 actúan en todas las coordenadas ALE (z\*, σ, híbrida).
- La pieza 2 solo actúa donde una celda se vaciaría.
- El σ euleriano (`vertical_coordinate="sigma_eulerian"`) no cambia.

No entran en este alcance:
- `visc_rem` de MOM6: es 1 sin fricción de fondo, y los casos de Ilicak no la tienen. Se agrega cuando un caso con fricción lo pida.
- La mezcla de HYCOM (1B): es un respaldo que requiere el OK de Esteban (§3).
- Un limitador en el remapeo: solo si la Tarea 0 lo pide, y se decide con Esteban.

## Criterios de éxito

**Obligatorios:**
1. **Onda interna híbrida con θ_bc = ½:** completa los 200 días con ν = 0.01, 15 y 150. Oscilación final / inicial de 0.65–0.75, 0.55–0.65 y 0.24–0.27.
2. **Overflow híbrido con ν = 1000:** completa las 40 h. Max\|u\| en las caras con un lado de menos de 1 cm queda por debajo del max\|u\| en las caras de al menos 1 m.
3. **Ninguna corrida ALE se detiene por "capa vaciada".** Lo garantiza la §2. Un error que quede tiene que ser otro (truncaciones o CFL barotrópico) y dar su causa.
4. **T acotada:** en todas las coordenadas ALE y en los cuatro casos de Ilicak, T queda dentro del rango inicial en celdas de al menos 1 mm (excursión ≤ 1e-6 °C).
5. **Sin regresión:**
   - lock exchange híbrido con RPE ≤ 2 veces la actual (8.8e5 J en `validation/mom6_ke_schemes`; 6.2e-8 a 7.5e-8 de ΔRPE/RPE₀ en el benchmark) y frentes de 0.96 a 1.04 U;
   - lock exchange z\* igual a MOM6 z\* (±5 % en RPE);
   - BayIW híbrido 2-D y corte 3-D dentro de las métricas validadas contra MITgcm (`validation/bayiw_2d`, `validation/bayiw_cut`);
   - las huellas guardadas de z\*/σ cambian solo por FCT, y eso queda documentado.
6. **Costo:** como mucho +15 % por paso en BayIW (560×352 y corte), en z\* y en la híbrida.

**Meta medida, que no bloquea:** el overflow híbrido con ν = 1 y 100. Si se logra, pyOCEAN supera a MOM6 ρ y a HYCOM. Si no, se documenta hasta dónde llega y por qué.

## §0. Tarea 0: diagnóstico (scratch, no entra al código)

En el overflow híbrido con ν = 100 (θ_bc = 0.6) y en la onda interna con θ_bc = ½:
- Se mide qué término de la tendencia acelera las caras con un lado de menos de 1 cm entre las 2 y las 5 h: gradiente de presión de volumen finito, gradiente de KE, vorticidad (q·(vh, −uh)), viscosidad horizontal o corrección barotrópica.
- Se mide dónde nacen los valores de T fuera de rango: advección horizontal (`_advect`) o remapeo vertical (PPM_H4).

**Resultado esperado:** el término dominante y su magnitud, con un renglón por término. Si el término dominante no es el que corrige la §3 (por ejemplo, un error del gradiente de presión en capas que afloran), la §3 se rehace con Esteban antes de implementarla. Si los extremos nacen en el remapeo, se decide con él un limitador en el remapeo.

## §1. Orden

Primero va el limitador (§2): garantiza que el modelo no se detenga mientras se ajusta §3, y es la condición de la que depende la garantía de §4.

1. Tarea 0
2. §2, limitador de salida
3. §3, acoplamiento de capas delgadas
4. §4, FCT
5. §5, θ_bc y restart
6. §6, validación y documentación

## §2. Limitador de salida (h ≥ 0 garantizado)

`limit_outflow(h, Qx, Qy, area, dt, active, h_floor=H_PPM_MIN) -> (Qx, Qy, n_limited)` en `core/vertical/ale.py`, con un kernel Numba y su oráculo NumPy. Se llama en `_step_ale` después de armar Qx y Qy (también en las caras abiertas) y antes de `lagrangian_thickness`. La advección de trazadores y Fz ya usan esos Qx y Qy.

**Algoritmo**, por capa k y celda c:
1. **Salida y disponible.**
   - Salida O = suma de los flujos salientes de las cuatro caras (x e y juntas, sin división direccional).
   - Disponible S = max(h_k − h_floor, 0)·A/Δt, con h_floor = 2e-10 m, el piso de PPM.
2. **Escalado.** Si O > S, los flujos salientes de (k, c) se multiplican por r = S/O. Cada cara sale de un solo donante, así que el escalado no es ambiguo.
3. **Reparto de lo recortado.** Lo recortado en cada cara (δ, con su signo) se reparte entre las otras capas de esa cara para que Σ_k Q_k no cambie:
   - **primero**, entre las capas que fluyen en el mismo sentido (mismo donante), en proporción a su capacidad restante S − O;
   - **lo que no quepa**, reduciendo las capas que fluyen en sentido contrario, en proporción a |Q|, sin cambiar su signo. Eso solo reduce la salida de la celda vecina, así que siempre es seguro.
4. **Repetición y error.** Se repite hasta que no quede ninguna (k, c) con O > S, con un máximo de 10 pasadas. Si aun así queda, el agua de la columna no alcanza para el flujo barotrópico: es el CFL barotrópico. Se lanza RuntimeError con la celda, Δt y el flujo.

**Propiedades:**
- Las velocidades no cambian (como en MOM6).
- Donde ninguna celda excede su disponible, Qx y Qy salen bit a bit iguales.
- `lagrangian_thickness` conserva su verificación como red de seguridad: si salta después del limitador, es un bug.
- `n_limited`, el número de (k, c) limitadas, se suma en el estado (`SigmaState.n_limited`) y se reporta con `n_trunc`.

**Tests:**
- una celda de 1 mm divergente con Courant 0.45 + 0.52 y corrección barotrópica da h† ≥ 0;
- Σ_k Q_k por cara se conserva a 1e-12 relativo;
- sin cambios cuando no hace falta (bit a bit);
- un caso que exige reducir capas en sentido contrario;
- Numba igual a NumPy;
- una columna sin agua lanza un error con su causa;
- la onda interna híbrida con θ_bc = ½ pasa del día 103, donde hoy falla.

## §3. Acoplamiento viscoso de las capas delgadas (1A)

**Hoy.** La viscosidad vertical implícita de la cara u recibe `hu = (q_E + q_W)/2`, con piso `DZ_FLOOR`. Una capa de 1 mm de un lado y 50 m del otro aparece con 25 m: casi no se acopla a sus vecinas.

**Cambio.**
- Solo para el acoplamiento viscoso se usa el grosor de cara de MOM6 (`MOM_vert_friction.F90:1457-1532, ~2236`, la definición por defecto, que la tarea del plan copia con file:line): la media armónica de los dos lados (o la combinación de MOM6) más h_neglect.
- Con 1 mm de un lado, el grosor es de ~2 mm, ν Δt/h² resulta enorme, y la capa queda atada a sus vecinas en la vertical.
- El transporte, el gradiente de presión, el remapeo de u y la regla de lecho delgado siguen con sus grosores actuales.

**Opción.** `thin_layer_coupling: str = "mom6" | "none"`. El valor por defecto es "mom6" en ALE si se cumplen los criterios 5 y 6. Si no, queda "mom6" solo en la híbrida y "none" en z\*/σ, con el motivo documentado.

**Respaldo (1B, solo con OK de Esteban):** la mezcla de HYCOM, u_k ← (u_k q + u_vecina (c − q))/c, con q = min(h_cara, c) y c = 0.5 m (`momtum.F90:3055-3062`).

**Tests:**
- una capa de 1 mm entre dos gruesas con velocidades distintas toma la velocidad de sus vecinas en un paso, con diferencia ≤ 1 % de la diferencia inicial;
- las capas gruesas cambian menos de 1e-6 m/s;
- en el overflow con ν = 1000, se cumple el criterio 2;
- regresión del lock exchange híbrido (criterio 5);
- Numba igual a NumPy si toca un kernel.

## §4. T acotada: FCT sobre WENO5-Z (Zalesak 1979, ec. 6–15)

**Dónde.** `tracer_limiter: str = "fct" | "none"` en `_advect`, el SSP-RK3 en forma de flujo con Q congelados y volumen que va de V0 a V1. Lleva un kernel Numba y su oráculo NumPy.

**Algoritmo:**
1. **Flujo de alto orden del paso:** F_H, la combinación de los flujos de los tres estados de SSP-RK3 (1/6, 1/6, 2/3), de modo que V1·c_RK3 = V0·c0 − Δt ∇·F_H. Un test comprueba esa identidad contra el `_advect` actual.
2. **Bajo orden:** c_L, con donor cell y los mismos Q, de modo que V1·c_L = V0·c0 − Δt ∇·(Q c_up). Con la §2, c_L es una combinación convexa de c0 y de los valores que entran: queda acotada sin excepciones.
3. **Límites:** c_max y c_min son los extremos de c0 y c_L en la celda y sus vecinas x, y de la misma capa (celdas activas). En las caras abiertas de entrada se incluye el valor exterior.
4. **Limitador.**
   - El flujo antidifusivo es A = F_H − F_L.
   - En cada celda, P± es la suma de los flujos antidifusivos que entran y salen, Q± = V1·(c_max − c_L) y V1·(c_L − c_min), y R± = min(1, Q±/P±).
   - Cada cara recibe C = min(R⁺ del receptor, R⁻ del donante) según el signo de A.
   - El resultado es V1·c = V1·c_L − Δt ∇·(C·A).
5. **Capas delgadas:** con V1 diminuto, Q± tiende a 0, C a 0, y la capa recibe upwind. No hace falta ningún umbral.

Se aplica igual a cada trazador de la pila: T, y T² para el diagnóstico de mezcla.

**Valor por defecto.** "fct" en ALE si se cumplen los criterios 4–6. Se espera que la RPE suba un poco, porque los extremos falsos la bajaban (Ilicak et al. 2012). Las huellas de z\*/σ ALE cambian:
- los tests que las fijan pasan a `tracer_limiter="none"` y deben reproducir las referencias viejas bit a bit;
- se agrega una huella nueva con FCT.

**Tests:**
- un escalón con Courant 0.9 queda en [0, 1] a 1e-12 con FCT, mientras que "none" se sale;
- una capa de 1 mm que recibe agua de otra T queda acotada;
- Σ V·T se conserva a round-off;
- en una sinusoide suave, el error con FCT es ≤ 1.5 veces el de "none";
- Numba igual a NumPy;
- en el overflow z\*, `t_excess` pasa de 4.7 °C a ≤ 1e-6 °C.

## §5. θ_bc y restart

- **θ_bc.** La regla automática pasa a θ_bc = ½ en todas las coordenadas ALE. Se quita el caso especial de la híbrida del constructor y del docstring, y se actualiza `test_baroclinic_theta_defaults_to_centred_except_hybrid`.
- **Restart:**
  - `_config` incluye `thin_layer_coupling` y `tracer_limiter`, y rechaza valores distintos;
  - `n_limited` se guarda como `n_trunc`;
  - un test comprueba que una corrida híbrida partida en dos con restart reproduce la corrida completa bit a bit.

## §6. Validación y documentación

Una corrida a la vez, con 5 hilos; las salidas van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`.

1. **Ilicak:** todas las corridas de pyOCEAN en z\*, σ e híbrida de los cuatro casos (`campaign.py results`, solo pyOCEAN), más la escalera de costo de z\* y de la híbrida en el overflow. Se aplican los criterios 1–5 y la meta.
2. **BayIW híbrido:** la sección 2-D y el corte 3-D contra MITgcm, con las métricas de sus README.
3. **z\* contra MITgcm:** `validation/internal_wave`, más BayIW 2-D y el corte. Estaban pendientes desde θ_bc y ahora cambian también por FCT.
4. **Costo por paso en BayIW** (560×352 y corte) para el criterio 6.

**Documentación:**
- README del benchmark: tablas, clasificación y la lista de mejoras, donde se cierran o actualizan los puntos 2, 3 y 4.
- `vision_roadmap.md` §7, CLAUDE.md y los docstrings de `ale.py` y `sigma_baroclinic.py`.
- `docs/literature/reading_list.md`: Zalesak (1979) ★, ya local; Hallberg & Adcroft (2009) y Higdon (2005, 2020), ya en la lista.

## Puntos de decisión con Esteban

1. Si la Tarea 0 muestra que la causa no es la que corrige la §3.
2. Si hace falta el respaldo de HYCOM (1B).
3. Si los extremos de T nacen en el remapeo.
4. Si un valor por defecto ("mom6", "fct") no cumple los criterios 4–6.

## Restricciones

- ≤ 5 hilos, una corrida pesada a la vez, sin núcleos de eficiencia y sin bajar la prioridad.
- No se modifica el código fuente de MOM6, HYCOM, MITgcm ni CROCO.
- Commits con rutas explícitas. Nunca se commitean `log_*.txt`, `*.npz` ni `pyOCEAN-literature-review-guide.docx`.
- Las salidas nunca van al repo.
