# N2a: salinidad, agua dulce real y balance exacto de varianza (validación: estuario de Lorenz et al. 2021)

**Fecha:** 2026-10-28. **Estado:** borrador para revisión de Esteban (revisión de constructor hecha; la pregunta abierta la contestó Esteban).
**Contexto:** etapa N de `docs/design/vision_roadmap.md` §8. Esteban decidió dividir N2 en dos partes:
- **N2a (este documento):** salinidad, ecuación de estado lineal en (T, S), evaporación y precipitación como flujo real de volumen, ríos y balance exacto de varianza;
- **N2b (después):** TEOS-10 (polinomio de Roquet) y flujos bulk de calor con onda corta penetrante.

Se construye sobre el núcleo de componentes de N1: `core/components/`, el driver de 10 fases y la matriz dorada `tests/test_n1_golden.py`.

## Por qué

El 3-D no tiene salinidad: el modelo es solo T, con una EOS lineal en T, y `step()` solo recibe viento. Sin S no hay estuarios, plumas de río, Golfo de California ni marea interna realista. El objetivo es el mejor modelo regional y costero (CLAUDE.md, "Project Goal"), así que la primera validación con sal se hace contra una **verdad exacta**, no contra un lugar.

Lorenz, Klingbeil y Burchard (2021, JPO 51, doi:10.1175/jpo-d-20-0158.1; `Lorenz_2021 - …md` en la biblioteca local) dan esa verdad. Extienden las relaciones de mezcla de Knudsen y TEF para incluir evaporación y precipitación, y su modelo GETM las cumple exactamente en un estuario 2-D idealizado (sus Fig. 2 y Tablas 1–2).

## Criterios de éxito (orden de prioridad)

1. **Cierre discreto exacto, criterio de aprobado o reprobado.** En el dominio cerrado y en cada paso, el balance de volumen, sal y varianza de salinidad cierra a redondeo, dividido en sus términos (frontera, superficie, mezcla física, mezcla numérica). Si no cierra, es error nuestro (clase a).
2. **Relación de Lorenz.** La mezcla total diagnosticada dentro del estuario (a la derecha del transecto) es igual a la de las relaciones (32)–(35) del artículo, evaluadas con TEF en el transecto y con el mismo promedio largo. Debe concordar como en su Tabla 2: "agrees exactly".
3. **Comparación con su Tabla 2.** Se comparan las magnitudes con GETM y cada diferencia se clasifica (a)–(d) según CLAUDE.md, sin ajustar nada al caso. Ya se espera una diferencia (b): nuestro GLS no tiene término de flotabilidad (`physics/mixing/gls.py:12`) y la difusividad de los trazadores es constante. Esa diferencia es el insumo medido de N3.
4. **Sin cambios donde no hay sal.** Con `salinity=False`, valor por defecto, los 1480 tests y la matriz dorada de N1 quedan bit a bit, y el costo por paso de los casos sin S no cambia más de 1 %.

## Alcance

**Dentro:**
1. **Salinidad como trazador activo** en los dos caminos (σ euleriano y ALE): estado, advección (mismo esquema y FCT que T), difusión K_h y K_v, esponjas, fronteras abiertas (`OBData.S`, que ya existe), remapeo ALE, restart y salida (`S` ya está en `diagnostics/output.py:FIELDS`). Va en los dos caminos porque el núcleo no debe tener un modo que ignore la sal sin decirlo.
2. **EOS lineal en (T, S):** ρ = ρ₀ − α(T − T₀) + β(S − S₀) (Vallis 2006, ec. 1.156), con `s_coef` (β, kg m⁻³ por g kg⁻¹) y `s_ref` (S₀).
   - Todos los consumidores de la densidad pasan a ρ(T, S):
     - el PGF jacobiano y de dos términos;
     - el PGF de volumen finito (`core/vertical/pgf.py`, hoy con los argumentos `t_coef`/`t_ref`);
     - `reference_profile`/`ReferenceProfile`;
     - la densidad de la regla híbrida;
     - `rho_ref="auto"`.
   - El caso de Lorenz tiene **T constante (α puede ser 0)**: ninguna fórmula puede dividir entre `t_coef`. Hoy la difusión de la anomalía y el FV PGF lo hacen.
3. **Evaporación y precipitación como flujo real de volumen** (Huang 1993; MITgcm `useRealFreshWaterFlux`, `model/src/integr_continuity.F:93,142`; MOM6 `applyBoundaryFluxesInOut`, `src/parameterizations/vertical/MOM_diabatic_aux.F90:692`).
   - **Entrada:** el campo de superficie `fw` = P − E [m s⁻¹, positivo hacia el océano] por celda y por paso, en `step(st, tau_x, tau_y, fw=None)`. El forzamiento es superficial y se pasa en cada paso, igual que el viento; `fw` también puede ser un campo fijo.
   - **Superficie libre:** recibe la fuente de volumen A·fw en la continuidad de η, incluida la conservación exacta del camino CG (`WD.step`, `core/dynamics/wetting_drying.py:300-343`).
   - **Trazadores:**
     - El volumen cambia y la sal no: el agua dulce que entra o sale lleva S = 0.
     - La T del agua que sale o entra se toma igual a la T de la capa superior, para que el calor no cambie por el agua dulce. N2b la reemplaza por la temperatura del flujo bulk.
     - En σ, el flujo por la interfaz superior deja de ser cero: Fz_top = −A·fw, con su valor de trazador.
     - En ALE, h† de la capa superior incluye fw·Δt.
   - **Capas que se vacían:** si la evaporación vaciaría la capa superior en un paso, se limita y se cuenta, igual que el limitador de salida de N1. Se reporta en `n_limited`, nunca en silencio.
4. **Ríos.** Se reutiliza la frontera abierta `'clamped'` con transporte exacto (`OBData.transport`, `core/dynamics/open_boundary.py:209-213`; probada en DOME) con S = 0 y T del río. No hace falta un mecanismo nuevo. Si un río en un borde no basta para el caso, el constructor lo dice antes de agregar otra cosa.
5. **Balance de varianza discreto exacto** (Klingbeil et al. 2014, doi:10.1016/j.ocemod.2014.06.001, biblioteca local; GETM lo usa en Lorenz 2021, l. 800-806 de la copia .md):
   - La variación de varianza de cada operador se diagnostica celda por celda: advección horizontal y vertical, difusión K_h, difusión K_v, remapeo ALE, esponjas y el término de superficie de E y P (ec. 43 del artículo).
   - Con eso, mezcla física + mezcla numérica + flujos de frontera + término de superficie = tendencia, a redondeo.
   - Vale para S y para T.
   - La advección cuenta como **un solo término** de mezcla numérica, el del operador 3-D completo (SSP-RK3 + FCT, con los flujos de cara efectivos del paso), sin separar horizontal y vertical. El cierre exacto no lo necesita, y separarlo obligaría a advección por direcciones, lo que rompería el bit a bit de T.
   - El balance por operador es **opcional y apagado por defecto** (`variance_budget=False`). El `mix_num` actual no cambia.
   - El `mix_num` actual (Burchard & Rennau 2008, advección de T²) se conserva como diagnóstico de compatibilidad si difiere, y se documenta en qué difiere.
6. **Benchmark** `validation/lorenz_estuary/`: definición del caso, runner y comparación con la Tabla 2.
   - **Análisis TEF.** Se calcula **dentro del modelo, en cada paso**: transportes de volumen, s y s² por bin de salinidad en el transecto, con los mismos flujos de cara que el balance de varianza, como GETM en el artículo (TEF "during the model run for each baroclinic time step", 250 bins de 0 a 40 g kg⁻¹). Solo así la igualdad de Lorenz es una identidad discreta. Es un diagnóstico opcional del modelo, no un posproceso.
   - **Corridas.** Los **cuatro experimentos A–D de su Tabla 1** (A sin marea, B–D con M2 de 0.5 m), en **dos configuraciones** (decisión de Esteban):
     - **σ euleriano + GLS** (k-ε sin término de flotabilidad, como hoy);
     - **ALE-σ con ν y κ constantes.**

     En total ocho corridas.
   - **Qué se compara.** El cierre exacto y la relación de Lorenz se exigen en las dos configuraciones, porque no dependen del cierre turbulento. La comparación con la Tabla 2 en las dos mide cuánto pesa la falta de flotabilidad en GLS, que es el insumo de N3.
   - **Costo.** Primero se mide el costo por corrida con el Δt estable de 60 capas σ en 5 m de agua. Si una corrida pasa de ~24 h de reloj en los 5 P-cores, se consulta a Esteban antes de cambiar capas o tiempo de integración.
7. **Pre-flight:** errores claros, por ejemplo `fw` sin `salinity=True`, S sin `s_coef`, o un río sin S. El registro de variables agrega `S` (y `closure` ya existe).

**Fuera** (N2b, N3 o después):
- TEOS-10 y la EOS no lineal;
- flujos bulk de calor y onda corta;
- el cierre estratificado con difusividad para trazadores (N3, lo que la comparación con la Tabla 2 va a motivar);
- partir `SigmaWetDry.step`, salvo lo mínimo para la fuente de agua dulce;
- agua dulce en los modelos 2-D de marea;
- el caso realista del Golfo Pérsico.

## Configuración del caso (de Lorenz et al. 2021, §4b)

- **Geometría:** canal 2-D (una fila) de 100 km, dx = 200 m (500 celdas), fondo lineal de 15 m en la frontera oceánica a 5 m en el extremo del río, **60 capas σ uniformes**.
- **Frontera oceánica:** S = 30 g kg⁻¹ constante; marea M2 de 0.5 m en η en los experimentos con marea; Flather + Orlanski + esponja (la configuración probada de pyOCEAN).
- **Agua dulce:** cada flujo (río, E integrada, P integrada) es 5.0 m³ s⁻¹.
  - Experimento A: E sobre la mitad interior, sin río, sin marea.
  - Experimentos B–D: las combinaciones de su Tabla 1 (E en el tercio medio, P en el tercio derecho, río).
- **Temperatura constante.**
- **Integración:** 500 ciclos de marea para llegar al estado cuasi periódico, luego un promedio de 100 ciclos (el artículo explica por qué no es periódico).
- **Lo que fija el builder después de leer el artículo:**
  - la EOS de GETM (β) para comparar magnitudes;
  - la viscosidad y difusividad de fondo;
  - el Δt;
  - la ubicación de los transectos (líneas discontinuas de su Fig. 2).

  Lo que el artículo no dé se marca **[no dado en el artículo]** y se elige el valor por defecto de GETM, con cita a su fuente si está disponible, o el más simple, sin ajustarlo para acercarse a la tabla.

## Decisiones tomadas en la conversación

- N2a antes que N2b (Esteban).
- El estuario idealizado de Lorenz es la primera validación (Esteban).
- Criterio: lo mejor para el modelo, es decir el cierre exacto como prueba dura más la comparación clasificada con la Tabla 2 (Esteban delegó; propuesta aceptada).
- Agua dulce real, no flujo virtual de sal (propuesta, sin objeción).
- El benchmark corre en las dos configuraciones, σ + GLS y ALE-σ con ν y κ constantes (Esteban, después de la revisión del diseño).

## Decisiones mías (revisar)

- **Cuatro experimentos, los de la Tabla 1,** no "con y sin marea" cada uno.
- **TEF dentro del modelo,** con los flujos de cara del paso.
- **Un solo término de mezcla numérica por advección.**
- **Balance de varianza opcional,** apagado por defecto.
- **Costo:** si una corrida pasa de ~24 h, se consulta antes de recortar.

- **S en los dos caminos (σ y ALE)**, no solo en ALE.
- **`salinity=False` por defecto**, para que todo lo existente quede bit a bit. Con `salinity=False` el modelo se comporta como hoy, con S implícita constante.
- **`fw` como argumento de `step()`**, como el viento, y no como un componente de forzamiento con su propio reloj. Ese diseño es de N4 (`Simulation`).
- **T del agua dulce = T de la capa superior** en N2a.
- **Ríos por la frontera `'clamped'` existente**, no como fuentes puntuales en el interior.
- **Con S activa, la difusión K_h de la anomalía respecto de ρ_ref** se aplica a T y S como campos (sin anomalía), y el pre-flight avisa si se piden `rho_ref` y `kh > 0` juntos. Invertir la EOS para separar T_ref y S_ref no está bien definido; se documenta como limitación.

## Referencias de código (leer antes de implementar)

- **Flujo real de agua dulce:**
  - MITgcm `model/src/external_forcing_surf.F:149,258-262,367`, `integr_continuity.F:93,142`;
  - MOM6 `MOM_diabatic_aux.F90:692-1459`, en particular `netMassIn`/`netMassOut` y el valor de trazador del agua que sale.
- **Varianza discreta:** Klingbeil 2014 (biblioteca local) y Lorenz 2021, ec. 43 y §4a.
- **TEF:** Lorenz et al. 2019, que cita el artículo. No está en la biblioteca; agregarlo a `docs/literature/reading_list.md` si se usa.
