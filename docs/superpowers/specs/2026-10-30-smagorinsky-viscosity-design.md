# Viscosidad horizontal de Smagorinsky (laplaciana, forma de tensor de esfuerzos)

**Fecha:** 2026-10-30. **Estado:** aprobado por Esteban el 2026-10-30: las seis decisiones de diseño ("me parece muy bien") y la regla de cierre del criterio 5 ("si"). Revisión de constructor hecha.
**Contexto:** después de N3 y del remapeo exacto en contenido (`main` en `4608943`), antes de N2b. Lecturas de referencia (con `archivo:línea`) en `docs/design/smag_refs/`:
- `mom6_hor_visc.md`: `MOM_hor_visc.F90` (Smagorinsky laplaciana y biharmónica, cotas, forma de tensor, fronteras, parámetros de los ejemplos);
- `croco_mitgcm_visc.md`: CROCO (`hmix_coef.F`, `uv3dmix_S.F`) y MITgcm (`mom_calc_visc.F`), más la viscosidad de nuestras corridas de referencia;
- `pyocean_current.md`: el `ah` actual, dónde se conecta un término nuevo, la evidencia de los benchmarks y la literatura local.

## Por qué

pyOCEAN solo tiene una viscosidad horizontal **constante** `ah`. Es un laplaciano escalar A_h ∇²u aplicado a cada componente, con deslizamiento libre y sin peso por grosor de capa (`core/components/momentum.py:204-215`, `core/dynamics/nonlinear.py:224`). Donde el flujo se acelera localmente no hay nada que amortigüe el modo de malla. Los benchmarks lo muestran:

- **DOME híbrido + GLS** (`validation/dome/README.md`): max|u| de 5.7 m/s contra 3.2 de MITgcm, 1072 eventos del limitador y un truncamiento CFL en el día 31.8 (cara de 1.3 cm, v = 8.96 m/s). El control con ν_v = 1e-5 aborta con 1034 truncamientos. El caso `kv_low` se dejó fuera porque MOM6 y MITgcm también se detienen sin Smagorinsky ni Jackson. Clase (b).
- **Lorenz σ + GLS** (`validation/lorenz_estuary/README.md:90-101, 349, 367`): con A_h = 0, el chorro de vaciante hace crecer un modo de 3-4 Δx y obliga a usar Δt = T_M2/746. Además el frente tiene oscilaciones de escala de malla. Clase (c), con la (b) "sin viscosidad adaptativa" detrás.

Ilicak et al. (2012) relacionan la mezcla espuria con el número de Reynolds de malla Re_Δ = UΔ/ν. Un Smagorinsky lo acota localmente sin amortiguar todo el dominio, que es justo lo que no puede hacer un `ah` constante.

## Criterios de éxito

1. **Operador correcto:**
   - con ν constante, la forma de tensor reproduce el decaimiento analítico de una onda de corte sinusoidal a O(Δx²);
   - la energía cinética nunca crece por el término (disipación ≥ 0 en cada paso, con y sin capas delgadas);
   - la cantidad de movimiento lineal se conserva en un dominio periódico (Σ h u invariante a redondeo).
2. **Coeficiente:** ν_Smag = C Δ² |S| (definiciones abajo) coincide con la fórmula de MOM6 a redondeo en una malla no uniforme de prueba, en los puntos h y q.
3. **Cota de estabilidad:** ν ≤ ν_max con la cota de MOM6 (abajo). Hay un contador `n_visc_limited` en `SigmaState` para los puntos limitados.
4. **Nada se rompe:**
   - con `smagorinsky=0` (por defecto), las 31 configuraciones doradas de N1 y el smoke de 9 casos son bit a bit iguales a `main`;
   - el restart de una corrida con Smagorinsky es bit a bit;
   - el costo por paso con el término apagado no cambia (≤ 1 %).
5. **DOME híbrido + GLS** (10 km, 50 días, `ah` = 50 de fondo como Legg) con Smagorinsky: **cero** truncamientos CFL y max|u| más cerca de MITgcm. Se reporta b_f, la trayectoria y el transporte contra la corrida sin Smagorinsky, clasificando cada cambio. **Si con C = 0.15 sigue habiendo truncamientos** (decisión de Esteban, 2026-10-30): se documenta la causa y el trabajo se cierra con este criterio abierto, como en N3. C = 0.5 se reporta como sensibilidad, no como arreglo, y el biharmónico no entra a este trabajo.
6. **DOME `kv_low`** (ν_v = 1e-4), solo en la coordenada híbrida y con `ah` = 50 de fondo, con Smagorinsky: completa 50 días o, si no, se documenta dónde y por qué se detiene. Es un caso que hoy no existe.
7. **Lorenz σ + GLS:** se prueba si Δt = T_M2/498 (el de ALE-σ) es estable con Smagorinsky, 150 ciclos en el experimento D (el que se detiene en el ciclo 131 con T_M2/498 en la búsqueda de 10a), comparado con la corrida existente de T_M2/746 en los mismos ciclos. El resultado se reporta sea cual sea. La relación exacta de Lorenz sigue cerrando (≤ 5e-10): la viscosidad solo toca la cantidad de movimiento.
8. **Ilicak:** el banco no cambia, porque Smagorinsky está apagado por defecto. Se agrega el caso de la Fig. 19(b) de Ilicak 2012: overflow con fondo ν_h = 100 m²/s más Smagorinsky C_s = 1/√2 (C = 0.5, el valor del artículo; es reproducir su caso, no calibrar), en z\*, comparado con nuestra corrida existente de ν_h = 100 sin Smagorinsky. Se espera la tendencia del artículo: menos mezcla espuria.
9. **Costo:** se reporta el costo por paso con Smagorinsky activado en BayIW cut z\* y en DOME 10 km. El objetivo es ≤ 10 % sobre el paso sin él; si se pasa, se documenta.

## Alcance

**Dentro:**
- Smagorinsky **laplaciana** en σ y en ALE (z\*, σ, híbrida), con `momentum='flux'` y `'vector_invariant'`.
- Forma de tensor de esfuerzos pesada por grosor, por capa, como `MOM_hor_visc.F90`. La tensión D_T y el corte D_S se calculan en las celdas (h) y en las esquinas (q).
- ν en h y en q por separado, como MOM6.
- Combinación con `ah` por **máximo**: ν = max(ah, C Δ² |S|). Cuando Smagorinsky está activo, el `ah` de fondo entra por el operador de tensor.
- Cota de estabilidad de MOM6 con el factor de capas delgadas, y el contador de puntos limitados.
- Deslizamiento libre en paredes de tierra; fronteras abiertas, periódicas y de mojado y secado.
- Pre-flight, restart, registro de variables (diagnóstico `nu_smag` en h, opcional en la salida), parámetro en `SigmaBaroclinic`.
- Kernel Numba con su oráculo NumPy, bit a bit para cualquier número de hilos.
- Tests y corridas de benchmark de los criterios 5–9; documentación (README de DOME y Lorenz, manual de momentum, CLAUDE.md, roadmap §7).

**Fuera** (pasos posteriores):
- Smagorinsky biharmónica, que usa el DOME de MOM6, y la cota repartida entre laplaciano y biharmónico.
- Leith y Leith modificado.
- Pared no-slip.
- Viscosidad dependiente de la resolución del radio de deformación (Hallberg 2013).
- Viscosidad o difusividad de trazadores tipo Smagorinsky.
- Mover el `ah` constante actual al operador de tensor cuando Smagorinsky está apagado: cambiaría las doradas sin ganar nada.

## Definiciones (de `MOM_hor_visc.F90`, ver `smag_refs/mom6_hor_visc.md`)

- Tensión en h: D_T = ∂u/∂x − ∂v/∂y. Corte en q: D_S = ∂v/∂x + ∂u/∂y.
- |S| en h = √(D_T² + media de los 4 D_S² de las esquinas). |S| en q = √(D_S² + media de los 4 D_T² de las celdas vecinas). La divergencia **no** entra, como en MOM6 y MITgcm (CROCO sí la incluye).
- Δ² = 2 dx² dy² / (dx² + dy²), la media armónica de dx² y dy².
- ν = max(ah, C Δ² |S|), con C = `smagorinsky` adimensional. 0 = apagado. El valor de MOM6 (`SMAG_LAP_CONST`, OM4p5) es 0.15.
- Cota: ν ≤ 0.8 · (6/11) / (4 Δt (1/dx² + 1/dy²)) · hrat_min, donde hrat_min ≤ 1 es la razón entre la cara vecina más delgada y el grosor local (forma de MOM6 `BOUND_KH`). Se aplica siempre que Smagorinsky está activo. MOM6 divide 0.8 · 0.25 / Δt entre el **máximo** de los dos términos métricos por dirección (≈ 2/dx² y 2/dy², `MOM_hor_visc.F90:3509-3513`), no entre su suma: en malla uniforme cuadrada es 0.1 dx²/Δt, que es 0.8 × el límite de Euler hacia adelante. La forma con la suma de pyOCEAN coincide en malla cuadrada y es más conservadora en celdas anisótropas. pyOCEAN extrapola el término con AB3, cuyo intervalo de estabilidad en el eje real negativo es λΔt ≤ 6/11, así que la cota se reescala: ≈ 0.055 dx²/Δt en malla uniforme. Con la de MOM6, el ν limitado sería inestable bajo AB3 por ≈ 1.5×. Decisión de Claude, ver abajo.
- Esfuerzos: σ_xx = ν h D_T en h y σ_xy = ν h_q D_S en q, con h_q la media de tipo armónico de MOM6, que se anula si una cara se anula. La tendencia es la divergencia del esfuerzo dividida entre (h_cara + h_neglect).
- Tierra: D_S = 0 en esquinas de tierra (deslizamiento libre). Frontera abierta: D_S = 0 en la esquina de borde y la tendencia es cero en las caras del borde abierto, como MOM6 (`smag_refs/mom6_hor_visc.md` §5).

## Restricciones

- Solo Python (NumPy y Numba); sin dependencias nuevas.
- Toda corrida larga exporta `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`, usa ≤ 5 hilos en los núcleos P, una corrida a la vez, y escribe en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`.
- Con `smagorinsky=0` el camino actual no cambia ni en el orden de las operaciones.
- Cada esquema y fórmula cita su fuente (artículo y `archivo:línea` de MOM6/CROCO/MITgcm) en el docstring.
- Sin ajustar el coeficiente a un caso. C = 0.15 es el valor por defecto de los benchmarks. Se reporta una sola corrida de sensibilidad con C = 0.5 (el C_s = 1/√2 de Ilicak 2012, Re_Δ ≈ 2) en DOME, documentada como sensibilidad, no como calibración.
- Los papers que se usen y no estén en la biblioteca local van a `docs/literature/reading_list.md`: Smagorinsky 1963, Griffies & Hallberg 2000 y cualquier otro que se cite.

## Decisiones de diseño (aprobadas por Esteban el 2026-10-30)

1. Laplaciana primero; la biharmónica queda para después.
2. Forma de tensor pesada por h, por capa (MOM6). El `ah` constante actual queda intacto cuando Smagorinsky está apagado.
3. C Δ² |S| con C = 0.15 por defecto en los benchmarks, Δ² armónica, combinación con `ah` por máximo.
4. Cota de MOM6 con el factor de capas delgadas y un contador.
5. Deslizamiento libre.
6. Apagado por defecto (`smagorinsky=0`).

## Decisiones tomadas por Claude (Esteban las revisa al leer)

- **Nombre del parámetro:** `smagorinsky` (float, adimensional) en `SigmaBaroclinic`, de forma análoga a `ah`.
- **Nivel de tiempo:** ν y los esfuerzos se calculan con el estado tⁿ, dentro de la tendencia explícita de la fase 2, extrapolada por AB3 como el resto de términos explícitos. Es lo que hacen CROCO y MITgcm (MOM6 usa la velocidad promediada de su RK2).
- **Lugar en el código:** un término con nombre en `Momentum.terms` (fase 2 del driver) que reemplaza a `"hviscosity"` cuando Smagorinsky está activo. No es un componente nuevo con rol propio.
- **σ:** se aplica a lo largo de las superficies σ, como el `ah` actual y como CROCO con `UV_MIX_S`. Sobre pendientes fuertes eso no es horizontal; se documenta sin corregirlo.
- **Mojado y secado:** las caras secas cuentan como tierra en ese paso (máscara `wet_u`/`wet_v` existente).
- **Cota de estabilidad reescalada para AB3** (revisión de constructor). La alternativa era sacar el término de AB3 y avanzarlo con Euler hacia adelante, como CROCO (`uv3dmix_S.F`). Se descartó porque cambia el lugar del término en el driver y en la historia del restart, y el `ah` actual ya va por AB3, igual que MITgcm (`momDissip_In_AB`). Si la cota reescalada limita demasiado en DOME (contador alto), se documenta.
- **Casos de benchmark:** DOME `kv_low` solo en la híbrida, con `ah` = 50 de fondo; Lorenz en D, a 150 ciclos con T_M2/498; Ilicak con el overflow de la Fig. 19(b), ν_h = 100 + C = 0.5, en z\*.

## Lo que decide el constructor (y lo documenta)

- La estructura exacta de los kernels: cuántas pasadas y cómo se fusionan.
- El halo periódico necesario, previsiblemente 2.
- El valor de `h_neglect` y su fuente, que debe ser el mismo que usa el resto de ALE.
- El caso analítico concreto del criterio 1, con su error esperado.
- Si `nu_smag` va a la salida por defecto o solo bajo pedido.
