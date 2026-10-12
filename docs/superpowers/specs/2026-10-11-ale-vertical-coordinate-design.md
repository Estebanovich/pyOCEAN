# Coordenada vertical general ALE (Lagrangiano vertical + remapeo): diseño

Aprobado por secciones con Esteban el 2026-10-11. Adelanta la etapa C de `docs/design/vision_roadmap.md`.

## 1. Motivación y criterio de éxito

**Problema.** En BayIW, la boca de la bahía es un escalón de 600 a 164 m con rx0 = 0.57 y rx1 = 684. ROMS recomienda rx0 < 0.2 y rx1 < 3–7. Con capas σ ese escalón genera un modo espurio:
- la varianza de T a 110 m en Seiche2 es de 59 mK, contra 3 mK en MITgcm;
- la varianza crece en el tiempo: 11 → 21 → 96 mK;
- el modo tiene un periodo de ~8 h e impide que el seiche siga decayendo después de 32 h.

Detalles en `validation/bayiw_iw/README.md`. MITgcm no tiene este problema porque usa niveles z con celdas parciales.

**Objetivo.** Un motor de coordenada vertical general que dé, con un solo algoritmo:
- z\* (escalones en z como MITgcm);
- σ (lo ya validado);
- una híbrida z\*–isopicna (la de menor mezcla espuria en MOM6 regional, Chang et al. 2026).

Es la base para la tesis (ondas internas, mezcla) y para comparar con HYCOM-IOA.

**Criterio de éxito (BayIW con z\* contra MITgcm):**
- varianza de T a 110 m en Seiche2 de ~3 mK;
- el seiche sigue decayendo después de 32 h y desaparece el residuo de ~8 h;
- tiempo por paso ≤ +25 % sobre el modelo σ (~3 s/paso);
- los 926 tests actuales siguen pasando (el modo σ actual queda intacto).

**Decisiones tomadas en el diseño:**
- enfoque A, Lagrangiano vertical + remapeo, como MOM6 y HYCOM, evolucionando el núcleo actual;
- primera versión con z\*, σ e híbrida;
- z̃ y las coordenadas adaptativas se evalúan después;
- validación contra cada modelo de referencia.

## 2. Arquitectura y orden del paso

**Estado.**
- `SigmaState` gana `h (N, ny, nx)`, el grosor de capa pronóstico en m (k = 0 en el fondo), con Σ_k h = H + η.
- u y v siguen siendo velocidades de capa en las caras; T y S, valores centrados en la capa.
- El restart guarda h, porque la malla depende de su historia.

**Paso de tiempo** (cada Δt baroclínico, conservando AB3 y la superficie libre implícita θ):

1. **Paso lagrangiano**, sin flujo a través de las interfaces (ω ≡ 0):
   - advección horizontal WENO5 de momento y trazadores, Coriolis, gradiente de presión, viento y fronteras;
   - transporte por capa aguas arriba (Plan B2, 2026-10-12; sustituye a min(h_L, h_R), que impedía a una capa isopicna entrar donde se había desvanecido): Q_k = L·u_k·q_k, con q_k el grosor de la celda de la que viene el agua;
   - en z\* e híbrida, q_k se recorta a la parte de la capa que está sobre el fondo más somero (HYCOM `cnuity.F90:252-256`); en σ no, porque las capas siguen el terreno. Sin recorte, una capa z\* frente a roca se acelera sin freno (seamount: 0.02 → 35 cm/s en un día);
   - Q_bt es el flujo de la superficie libre implícita; el residuo Q_bt − Σ_k Q_k se reparte en proporción a los q_k de la celda aguas arriba del residuo (HYCOM `cnuity.F90:585-620`), así que Σ_k Q_k = Q_bt;
   - el grosor de cara independiente de la dirección, (q_E + q_W)/2, define ū = Σ w_k u_k, la viscosidad y la regla de capas de fondo delgadas;
   - **Plan B2b (opcional, §9):** q_k con PPM y limitador de positividad integrado sobre el volumen barrido, y ajuste de velocidad uniforme por Newton para cerrar Σ_k Q_k = Q_bt (MOM6 `MOM_continuity_PPM.F90:1113-1250, 1368-1546`; Adcroft et al. 2019, §A). Se adopta solo si afila el frente del lock exchange híbrido (§7);
   - continuidad de capa: h_k† = h_k − Δt ∇·Q_k;
   - como η usa el mismo Q_bt, Σh† = H + η^{n+1} por construcción. Es la misma consistencia que imponen MOM6 (`zonal_flux_adjust`, `MOM_continuity_PPM.F90:1368-1546`) y CROCO (`step3d_uv2.F:924-1018`);
   - advección de momento en forma de flujo con limitador en celdas desvanecidas: V_u ≥ 2Δt·Q_entrada, estable con AB3 (λΔt < 6/11); mismo propósito que el truncamiento CFL de MOM6 (`MOM_vert_friction.F90:2555`). Sin él, una capa de 1 cm inundada por la corriente llegaba a −84 m/s.
2. **Física vertical** sobre la malla lagrangiana: viscosidad y difusión implícitas, viento como flujo en el tope y GLS. Va antes de regenerar la malla, como en MOM6 (`MOM.F90:1774` → `:1033`) y HYCOM (`mod_hycom.F90:2584-2650`).
3. **Regenerar la malla** con la regla elegida (§3):
   - no cambia Σh (`MOM_regridding.F90:1384`);
   - el grosor mínimo es min(1 mm, H/N) y se infla desde el fondo.
4. **Remapear** de h† a la malla nueva (§4):
   - T, S y los trazadores, con conservación al redondeo;
   - u y v con el grosor en la cara igual al promedio de las celdas vecinas, para la malla vieja y la nueva (`MOM_ALE.F90:925-931`);
   - en capas de fondo de menos de ~1 cm se fuerza velocidad total cero (HYCOM, `hybgen.F90:1757-1764`).
   - Frecuencia: cada paso, como HYCOM; opción de cada n pasos, como `DT_THERM` de MOM6.

**Código.**
- `core/vertical/remap.py`: reconstrucciones y remapeo por columna (Numba, prange sobre columnas).
- `core/vertical/regrid.py`: reglas z\*, σ e híbrida.
- `core/vertical/pgf.py`: gradiente de presión de volumen finito (§5).
- `core/vertical/coordinates.py`: configuración y validación de parámetros.
- `SigmaBaroclinic(vertical_coordinate="sigma_eulerian" | "ale", regrid=<regla>, remap_scheme=...)`.
- El modo `"sigma_eulerian"` (actual) es el valor por defecto y no cambia.

## 3. Reglas de malla

Cada regla es una función pura por columna: recibe (h†, T, S, H, η) y devuelve las interfaces objetivo.

1. **z\*** (Adcroft & Campin 2004; `coord_zlike.F90:91-143`):
   - grosores nominales Δz_k (para BayIW, `delR` de MITgcm) escalados por (H + η)/H;
   - las capas bajo el fondo colapsan a h_min en el fondo, con un barrido hacia arriba;
   - la capa activa más profunda queda truncada, que equivale a la celda parcial de MITgcm.
2. **σ** (fracciones fijas: uniformes, Song & Haidvogel o `delR`; `coord_sigma.F90:68-83`). Reproduce el modelo actual.
3. **Híbrida z\*–isopicna**, tipo HYCOM1 de MOM6 (`coord_hycom.F90:161-280`; OM4, Adcroft et al. 2019):
   - densidad de las interfaces con el truco de compresibilidad, p = p_ref + f_c (ρ₀ g z − p_ref) (`MOM_regridding.F90:2062-2063`);
   - las capas sin masa (h < 2 h_min) toman, solo para la regla, la densidad de la capa con masa de arriba (de abajo si no hay ninguna), como el relleno de HYCOM (`hybgen.F90:904-925`). Sin esto, una capa de fondo de 1 mm con el agua ligera inicial aligera toda la columna al forzarla monótona y disuelve una corriente de gravedad (Plan B2, 2026-10-12);
   - perfil de ρ forzado a ser monótono (`coord_hycom.F90:218`);
   - reconstrucción P1M y búsqueda de Newton de cada densidad objetivo dentro de su celda (`regrid_interp.F90:378-511`);
   - **piso z\*:** una interfaz no queda por encima de su posición z\* (`coord_hycom.F90:255-267`);
   - **grosor máximo por capa** (`MAX_LAYER_THICKNESS`), por la falta de resolución en zonas poco estratificadas que reportan Chang et al. (2026);
   - **paso a σ en aguas someras por ventana** (decisión del 2026-10-11, sustituye a la mezcla de posiciones del Plan A):
     - q = clip((D − d_σ)/(d_hyb − d_σ), 0, 1);
     - cada interfaz k queda entre un límite superior U_k = q·z\*_k + (1 − q)·σ_k y uno inferior L_k = q·(−H) + (1 − q)·σ_k, es decir, z_k = clip(z_k^híbrida, L_k, U_k), y después se aplica el grosor mínimo;
     - q = 0 da σ pura (mojado-secado); q = 1 da la híbrida sin cambios; en medio, las isopicnas que caben en la ventana se respetan (12 de 19 interfaces a q = 0.5 en una columna de estratificación lineal, contra 0 con la mezcla de posiciones);
     - es continua en D (que cambia con la marea), las interfaces no se cruzan y Σh = D;
     - HYCOM hace otra cosa (`MOM_hybgen_regrid.F90:590-620`): mezcla los mínimos dp0k/ds0k con un peso que hace que sumen D, así que entre dsns y dpns la columna es σ pura y no sigue isopicnas; la ventana da una transición gradual;
   - el paso a σ también define dónde se permite el mojado-secado;
   - función auxiliar para elegir las densidades objetivo desde el perfil inicial, con la opción de objetivos más ligeros que el agua superficial en las capas de arriba (estilo OM4) para que la capa de mezcla quede resuelta en z;
   - filtro temporal de la malla opcional y apagado por defecto (`MOM_regridding.F90:1461-1608`).

**Conservación de las capas isopicnas** (Plan B2c, aprobado el 2026-10-12). En flujo adiabático, la regla anterior movía las interfaces en cada paso hacia donde la reconstrucción P1M cruza las densidades objetivo. Ante un salto brusco de densidad las repartía entre las capas intermedias, y el remapeo mezclaba (lock exchange híbrido, 6 h: el 92 % de la mezcla espuria viene del remapeo; ganancia de RPE 3.1e8 J, igual a ALE-σ).
- **Regla:** la capa j (contada desde arriba, entre los objetivos t_j y t_{j+1}) es *consistente* si no tiene masa (h < 2 h_min) o si su densidad está en su rango, t_j ≤ ρ_j ≤ t_{j+1}. Una interfaz entre dos capas consistentes se queda donde la dejó el paso lagrangiano; las demás se colocan con P1M como antes.
- **Orden:** una interfaz que se recoloca se acota entre las conservadas más cercanas de arriba y de abajo, así que nunca empuja a una conservada. El piso z\*, el grosor máximo, la ventana σ y el grosor mínimo se aplican después, igual que ahora, y sí pueden mover cualquier interfaz.
- **Parámetro:** `keep_isopycnal` (True por defecto en la regla híbrida). False reproduce la regla anterior, que es la que se compara con MOM6.
- **Fundamento:** las capas isopicnas puras no tienen difusión espuria y la coordenada continua sí (White et al. 2009, figs. 11–12). `hybgen` solo mueve las interfaces de capas que se alejan de su objetivo (Bleck 2002). Aquí se toma ese principio sin portar `hybgen`: las capas inconsistentes se siguen recolocando por interpolación, no por arrastre de primer orden.
- **Medido en un prototipo:** ganancia de RPE 2.5e7 J (12× menos). Una reconstrucción discontinua (PLM o PPM_H4, la otra vía de White 2009) empeoró la mezcla (4.3–5.2e8 J), porque las capas desvanecidas con valores viejos deforman las pendientes.
- **Lo que queda:** una película densa delgada se adelanta a la corriente (frente medido a 1 m: 1.75 U). Viene del transporte de grosor de primer orden, no de la regla, y es lo que debe resolver B2b.

**`hybgen` no se porta** (decisión del 2026-10-11, tras la comparación de `validation/mom6_column/`). Con objetivos ligeros arriba, la regla híbrida coincide con `hybgen` en el interior (diferencia media de 4.0 m en BayIW y 0.8 m en la columna estratificada, ≤ 0.4 grosores de capa locales). Las diferencias mayores vienen de dos rasgos de `hybgen`: capas z superiores fijas (`fixlay`) y capas vacías alternadas, el "thin-thick-thin null mode" que reconoce el propio MOM6 (`MOM_hybgen_regrid.F90:806-813`). En su lugar se mejora la regla propia: objetivos ligeros arriba, grosor máximo por capa y la transición por ventana.

## 4. Remapeo

- **Reconstrucciones**, en este orden de implementación:
  1. PCM;
  2. PLM (pendiente de White & Adcroft 2008, `PLM_functions.F90:22-68`);
  3. PPM_H4 (por defecto);
  4. PQM_IH4IH3.
- **Limitadores:** valores de borde acotados (`regrid_edge_values.F90:41-103`) y luego Colella–Woodward (`PPM_functions.F90:57-125`).
- **Integración:** sobre la unión de las mallas de origen y destino (`MOM_remapping.F90:646-802`), con el truco de Hallberg: la sub-celda más gruesa absorbe el residuo y el contenido se conserva al redondeo (`:942-961`).
- **Bordes de la columna:** las celdas de arriba y de abajo usan PCM, sin extrapolación (`REMAP_BOUNDARY_EXTRAP=F`).
- **Columnas cortas:** se baja el orden (≤ 1 capa → PCM, ≤ 3 → PLM; `:453-460`).
- **Capas vacías:** `h_neglect` subredondeo en los denominadores; no se infla h durante la corrida.

## 5. Gradiente de presión y capas delgadas

- **Gradiente de volumen finito** de Adcroft et al. (2008), siguiendo `int_density_dz_generic_plm` (`MOM_density_integrals.F90:420-871`):
  - T y S con perfil PLM dentro de la capa y extrapolación en los bordes;
  - regla de Boole con 5 puntos en la vertical y 5 sub-columnas a lo largo de la capa, exacta para EOS lineal;
  - fuerza final según `MOM_PressureForce_FV.F90:1796-1800`;
  - se mantiene la resta del perfil de referencia ρ_ref(z), validada en el seamount.
- **Capas delgadas junto a topografía abrupta:** ponderación por masa (`MASS_WEIGHT_IN_PRESSURE_GRADIENT`, `MOM_PressureForce_FV.F90:2124-2149`; `MOM_density_integrals.F90:652-681`). Si una capa se extiende bajo el fondo de su vecina, T y S de las esquinas se toman de la celda más gruesa. Es el caso de la boca de BayIW con z\*.
- **Advección horizontal:** flujos L·q·u con q aguas arriba (§2), que toleran h → 0.
- **Viscosidad y difusión implícitas:** Thomas con dz muy pequeño, sin cambios.

## 6. Integración con el resto del modelo

### 6.1 Fronteras abiertas

Revisión de código de MOM6, HYCOM y CROCO, y de literatura. Ningún modelo resuelve del todo las fronteras con capas móviles, así que se toma lo mejor de cada uno y se evitan sus fallas documentadas.

1. **Datos externos en su propio eje z con grosores dz** (como `dz_<var>` de MOM6, `MOM_open_boundary.F90:1188-1204`):
   - en cada paso se ajustan al fondo y se remapean (PLM, conservativo) a los grosores *actuales* de la celda interior (`:4636-4825`);
   - se verifica que Σdz de la fuente coincida con la columna;
   - se evita la falla `OBC_TRACER_DZ_BUG` de MOM6 (`:778-781`), que remapea sobre grosores viejos.
   - T_ext deja de vivir en capas σ.
2. **Flather** dentro de la superficie libre implícita, con ū_ext = Σ(u_ext·h_int)/Σh_int (`MOM_barotropic.F90:4428-4458`).
3. **Transporte por capa en la cara abierta:** grosor de la celda interior (`MOM_continuity_PPM.F90:1237-1248`), con Q_k como en §2 para que Σ = Q_bt.
4. **Orlanski por capa:**
   - se radia la velocidad de la capa con el mismo índice (MOM6 `MOM_open_boundary.F90:2950-2972`; CROCO `u3dbc.F:112-160`);
   - nudging con escalas τ_in y τ_out elegidas por la dirección del flujo de información (MOM6 `:3014-3023`; Marchesiello et al. 2001).
5. **Todo estado auxiliar de la frontera** (velocidad de fase, reservorios) se remapea siempre al regenerar la malla. En MOM6 esto depende de `REMAP_AUXILIARY_VARS`, que está apagado por defecto (`MOM.F90:2588-2592`).
6. **Trazadores: reservorio de MOM6** (`MOM_open_boundary.F90:6150-6272`):
   - actualización implícita con el transporte que cruza la cara y longitudes L_in, L_out;
   - con L = 0 es el "valor aguas arriba" actual, que queda como caso particular y prueba de regresión;
   - con L > 0 se suaviza la entrada (Marsaleix et al. 2006).
7. **Esponja** (MOM6 `MOM_ALE_sponge.F90:1057-1066` + HYCOM `thermf.F90:816-858`):
   - objetivos en z, remapeados en cada llamada a las capas actuales;
   - Euler implícito sobre T, S y la parte baroclínica de u y v; h no se relaja;
   - después de relajar u y v se quita la media vertical;
   - se conserva el perfil de MITgcm para BayIW.

**Fuera de alcance:** la frontera por relaciones de polarización para ondas internas (Marsaleix et al. 2009b). Entra después de medir la reflexión (λ = F_r/F_in, Siyanbola et al. 2023; pieza 8 de `docs/design/unified_core.md`).

### 6.2 Resto

- **Difusión horizontal:** sobre T − T_ref(z) a lo largo de las capas, válida para σ, z\* e isopicnas inclinadas.
- **Viento, viscosidad y difusión vertical, GLS:** por columna con h.
- **Kernels de advección Numba:** cambian D·Δσ por h en las caras; Fz = 0 en el paso lagrangiano.
- **Mojado-secado:** solo si la regla tiene zona σ somera que cubra las celdas que se secan; si no, el constructor da un error con una explicación.
- **Restart:** h, el estado del filtro temporal si existe, el estado auxiliar de la frontera, y la verificación de configuración (coordenada, regla, parámetros).
- **Salida CF:**
  - con σ, la fórmula `ocean_sigma_coordinate`;
  - con z\* o híbrida, h y z_w como coordenadas auxiliares (como `h` y `e` de MOM6);
  - la salida remapeada a z queda fuera de alcance.
- **Diagnósticos:** RPE y varianza de mezcla espuria, sin cambios. Miden cuánto mezcla el remapeo frente al WENO vertical.
- **Verificador de dominio:** rx1 con las capas reales (hoy asume σ uniforme) y recomendación de z\* o híbrida cuando rx1 es alto.

## 7. Pruebas (TDD)

**Unitarias:**
- **Remapeo:**
  - conservación al redondeo;
  - identidad con la misma malla;
  - exacto para polinomios del orden de cada esquema;
  - sin extremos nuevos;
  - capas vacías;
  - orden de convergencia (White & Adcroft 2008).
- **Reglas de malla:**
  - Σh preservado y grosor mínimo;
  - z\* = `delR` escalado;
  - σ = sus fracciones;
  - híbrida: interfaces en sus densidades objetivo (columna con estratificación lineal), piso z\*, paso a σ somero, grosor máximo;
  - ventana somera: σ pura con D ≤ d_σ, híbrida sin cambios con D ≥ d_hyb, interfaces en su isopicna dentro de la ventana con q intermedia, continuidad en D y orden de las interfaces.
  - capas sin masa: su densidad vieja no aligera la columna (una corriente densa de 0.7 m sobre una capa de fondo de 1 mm con agua ligera conserva más del 90 % de su agua);
  - conservación de capas isopicnas: con todas las capas consistentes las interfaces no se mueven (exacto); una capa inconsistente se recoloca como con P1M; una interfaz recolocada no cruza a una conservada; el piso z\* y el grosor mínimo siguen valiendo; `keep_isopycnal=False` reproduce la regla anterior bit a bit.
- **Transporte por cara:** grosor aguas arriba; recorte al fondo somero en z\* (capa frente a roca: 0; capa parcial: la parte sobre el fondo, como hFacW de MITgcm); suma igual al flujo de la superficie libre; ninguna capa vacía recibe el residuo.
- **Gradiente de presión:**
  - fuerza cero en reposo estratificado con capas inclinadas arbitrarias (EOS lineal);
  - error del seamount ≤ al actual.
- **Fronteras:**
  - remapeo de datos en z;
  - reservorio con L = 0 igual al valor aguas arriba;
  - nudging con τ_in y τ_out.

**Integración:**
1. Regresión de ALE con regla σ contra el modelo σ actual (canal modo 1, lock exchange), dentro de una tolerancia.
2. **Escalón 600 → 164 m** en reposo estratificado con una perturbación pequeña:
   - con σ crece la varianza espuria, lo que documenta el problema;
   - con z\* debe quedar al nivel de redondeo.
3. Fronteras: los canales actuales (reflexión 0.82 cerrado / 0.157 Orlanski / 0.015 esponja) con z\* e híbrida.
4. Restart bit a bit con ALE.
5. Mojado-secado con zona σ somera.
6. **Lock exchange híbrido** (cada masa de agua en una capa desvanecida al otro lado de la compuerta, 10 capas, 6 h):
   - el cuerpo de la corriente (5 m de los 10 m de Benjamin) avanza a la velocidad de Benjamin ±10 %;
   - con la conservación de capas isopicnas, la ganancia de RPE es < 0.2 veces la de ALE-σ;
   - con B2b, el frente medido a 1 m queda dentro de ±10 %; si no, B2b no se adopta y queda el xfail con los números medidos;
   - sin regresión: escalón z\* ≤ 1e-10 m/s, seamount z\* + volumen finito ≈ 0.1 cm/s a 10 días, tests σ sin cambios; volver a correr la comparación con `hybgen`.

## 8. Matriz de validación contra los modelos de referencia

**Nivel 1: componente contra el Fortran original, a redondeo.** Programas Fortran pequeños que llaman a las rutinas por columna originales:
- **Remapeo** PLM, PPM_H4 y PQM contra `remapping_core_h` de MOM6, más los valores esperados de `test_MOM_remapping.F90`.
- **Reglas de malla:** HYCOM1 contra `coord_hycom.F90`; z\* contra `coord_zlike.F90`; σ contra `coord_sigma.F90`.
- **Gradiente de presión** contra `int_density_dz_generic_plm` con EOS lineal.
- **HYCOM:** nuestra regla híbrida contra `hybgen` en columnas típicas (estratificada, mezclada, somera, capa de fondo ligera). Es una comparación cuantitativa de la posición de las interfaces, no a redondeo, porque los algoritmos difieren. Hecho en el Plan A: las diferencias no justifican portar `hybgen` (§3).

**Nivel 2: casos idealizados contra cada modelo completo** (con z\*, σ e híbrida en pyOCEAN):

| Caso | MITgcm | CROCO | MOM6 | Valida |
|---|---|---|---|---|
| Lock exchange (Ilicak et al. 2012) | ✓ | ✓ (corrida existente) | ✓ | mezcla espuria (RPE) |
| Seamount | — | ✓ | ✓ | error del gradiente de presión |
| Ondas internas (MITgcm `internal_wave`, CROCO IGW/INTERNAL, MOM6 `adjustment2d`/`sloshing`) | ✓ | ✓ | ✓ | dispersión y amortiguamiento |
| Overflow (MITgcm `dome`, CROCO OVERFLOW, MOM6 `flow_downslope` en sus variantes z/σ/ρ/capa = Ilicak et al. 2012, y `DOME`) | ✓ | ✓ | ✓ | flujo denso sobre pendiente |
| Escalón 600 → 164 m | ✓ | — | ✓ | el problema de la boca de BayIW |

- MOM6 se compila con FMS 2025.02.01 (`~/pyOCEAN_refs/FMS`) y las entradas de `~/pyOCEAN_refs/MOM6-examples/ocean_only/` (`lock_exchange`, `seamount`, `flow_downslope`, `DOME`, `adjustment2d`, `sloshing`), descargadas con aprobación el 2026-10-11.
- Cada caso sigue las reglas de benchmark de `CLAUDE.md`: clasificar las diferencias (a)–(d), escribir la lista de lo que debe mejorar pyOCEAN y actualizar la tabla de §7 del roadmap.

**Nivel 3: casos realistas.**
- **BayIW** contra MITgcm con z\* (criterio del §1) y con la híbrida.
- **HYCOM-IOA:** comparación de la etapa C cuando exista la frontera desde un modelo padre (pieza 7 de `unified_core.md`).

## 9. Orden de implementación

Cada fase se cierra con sus tests y su validación de nivel 1 o 2:

1. Remapeo (+ validación contra MOM6).
2. Reglas de malla (+ validación contra MOM6 y comparación con `hybgen`).
3. h pronóstico y paso lagrangiano; regresión ALE-σ.
4. Gradiente de presión de volumen finito; test del escalón; seamount.
4b. Plan B2 (2026-10-12): transporte aguas arriba con recorte (B2a, hecho), conservación de capas isopicnas (B2c), luego PPM en el transporte de grosor (B2b) si el lock exchange híbrido lo justifica.
5. Fronteras: datos en z, reservorio, nudging, esponja, estado auxiliar.
6. BayIW con z\* contra MITgcm.
7. Regla híbrida; casos idealizados de nivel 2; BayIW con la híbrida.
8. Restart, salida, verificador de dominio, documentación y tabla de §7.

## 10. Fuera de alcance (evaluar después)

- `hybgen` exacto: descartado el 2026-10-11 (§3);
- z̃ (Leclair & Madec 2011) y coordenadas adaptativas (Hofmeister et al. 2010; Chen et al. 2024);
- frontera por relaciones de polarización (Marsaleix et al. 2009b);
- salida remapeada a niveles z;
- barotrópico split-explicit.

## Referencias

Las locales están en `/Users/estebancruz/Documents/TESIS_InternalWaves-2/Literatura`; la lista con DOI está en `docs/literature/reading_list.md` §2 y §8.

**Coordenada vertical y remapeo:**
- Adcroft & Campin 2004; Adcroft & Hallberg 2006; Adcroft et al. 2008; Adcroft et al. 2019 (OM4).
- Bleck 2002; Griffies et al. 2020.
- White & Adcroft 2008; White, Adcroft & Hallberg 2009.
- Chang et al. 2026; Bruciaferri et al. 2024; Petersen et al. 2015.
- Transporte de capas: Hallberg & Adcroft 2009 [verificar]; Higdon 2005 (por descargar).

**Fronteras abiertas:** Marchesiello et al. 2001; Marsaleix et al. 2006, 2009b; Siyanbola et al. 2023.

**Código fuente de referencia:**
- MOM6 en `~/pyOCEAN_refs/MOM6/src/` (commit 0e9a561, 2026-10-07);
- HYCOM en `~/pyOCEAN/HYCOM-src/`;
- CROCO en `~/pyOCEAN_refs/croco/`.
