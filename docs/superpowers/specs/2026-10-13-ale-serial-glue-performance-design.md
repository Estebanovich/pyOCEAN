# Rendimiento del paso ALE, segunda parte: el exterior anidado y el pegamento serie

**Fecha:** 2026-10-13. **Estado:** diseño aprobado por Esteban en la conversación; falta revisar el texto escrito.
**Rama:** `ale-perf-2`, desde `main` (c8e929c).
**Antecedente:** `2026-10-12-ale-step-performance-design.md` (núcleos Numba guiados por el perfil, enfoque A). Esta parte sigue el mismo enfoque y las mismas reglas.

## 1. Diagnóstico (corte BayIW 198 × 159, dos capas con bahía, Δt = 120 s, 5 hilos, 50 pasos tras 5 de calentamiento)

**Costo por hora simulada:** MITgcm 4.4 s (5 procesos MPI, Δt = 30 s); pyOCEAN z\* 11.7 s (0.39 s/paso); la híbrida, de 25 capas, 6 s (0.20 s/paso); z\* anidado 15.6 s (0.52 s/paso).

**Escalado con hilos (z\*):** 0.76 s/paso con 1 hilo y 0.39 con 5, un factor de 1.97. Por la ley de Amdahl, ~38 % del trabajo de un hilo es serie. Con 5 hilos esa parte es **~75 % del tiempo de pared**: el freno es el código NumPy entre los núcleos, no los núcleos.

**Reparto (z\* radiando, % del paso, acumulado y con solapes):**

| Pieza | % |
|---|---|
| Tendencias de momento (`_tendencies`) | 18 |
| Advección de T (`_advect`) | 14 |
| Remapeo de T (`remap_columns`) | 13 |
| PGF FV (`fv_pgf`) | 11 |
| Remapeo de u, v en caras (`_remap_faces`) | 10 |
| Superficie libre (CG + LU: 2 %) | 8 |
| Flujo de grosor PPM | 8 |
| Espesores en caras (`_mean_face_thickness`, `_transport_thickness`) | 13 |
| `np.ascontiguousarray` (167 llamadas por paso) | 8 |
| `nan_to_num` (22 llamadas por paso) | 5 |

**Anidado:** `_evaluate_exterior` y `_exterior_T` suman el **28 %** del paso anidado.
- `exterior_tracer` (`core/dynamics/open_boundary.py`) copia el perfil de cada borde a arreglos del dominio completo (`_source_columns`, (nz, ny, nx)) y los remapea en toda la banda de esponja de 30 celdas en cada paso.
- Con `sponge_tracer=False` (el corte y BayIW), de esa banda solo se usa la T de entrada en la columna del borde.

## 2. Enfoque

**Parte 1: el exterior anidado.**
1. Calcular T exterior solo en las columnas que la usan: las del borde, más la banda cuando `sponge_tracer=True`.
2. Construir las columnas de origen solo en esos puntos (reunir y dispersar), sin arreglos del dominio completo.
3. Lo mismo para `exterior_velocity` y las metas de Flather, si el perfil lo justifica.

**Parte 2: el pegamento serie**, con el enfoque A del plan anterior:
- se mide, se reescribe la pieza serie más cara como núcleo `numba.njit(parallel=True, cache=True)` o se fusiona con el núcleo vecino, y se repite;
- la versión NumPy actual queda como oráculo en los tests;
- **candidatas, por el perfil:**
  - el cuerpo NumPy de `_step_ale` y `step`: máscaras, `np.where`, sumas por columna;
  - `_mean_face_thickness` y `_transport_thickness`;
  - la preparación de `_tendencies` y `_advect`;
  - las copias `ascontiguousarray`, que se evitan creando los arreglos ya contiguos;
  - `nan_to_num` sobre H y las profundidades de cara, que se calculan una vez al construir el modelo.
- **Criterio de parada:** el escalado de 1 a 5 hilos llega a ≥ 3.5 veces, o ninguna pieza serie pasa del 5 % del paso.

**Descartados:**
- reescribir el paso entero en un núcleo: difícil de probar;
- MPI o descomposición en teselas: es otra etapa;
- JAX: es la etapa D.

## 3. Metas (corte BayIW, 5 hilos)

| Caso | Hoy | Meta |
|---|---|---|
| z\* anidado | 0.52 s/paso | ≤ z\* radiando + 5 % |
| z\* radiando | 0.39 s/paso | ≤ 0.25 s/paso (~1.7 veces MITgcm por hora simulada); deseable 0.15 |
| híbrida radiando | 0.20 s/paso | ≤ 0.13 s/paso |

Si una meta no se alcanza, se reporta el reparto final y por qué.

## 4. Reglas

Son las del plan anterior.
- **Hilos y precisión:** ≤ 5 hilos; float64; sin `fastmath`.
- **Modo σ:** idéntico bit a bit.
- **Parte 1:** idéntica bit a bit en el modelo completo. Solo deja de calcular columnas que no se usan.
- **Parte 2:**
  - cada núcleo coincide con su oráculo NumPy (≤ 1e-12 relativo, o idéntico si hace las mismas operaciones en el mismo orden);
  - las huellas del modelo completo (`tests/data/ale_perf_reference.json`, `sigma_obc_reference.json`) se mantienen; si una cambia por el orden de una suma, se documenta la diferencia (≤ 1e-10 relativo) antes de regenerarla.
- **El restart bit a bit (Plan D) sigue pasando.**
- **Sin dependencias nuevas.**

## 5. Pruebas

- **Por pieza:** núcleo contra oráculo NumPy, en casos aleatorios con capas desvanecidas, tierra, bordes y caras abiertas.
- **Parte 1:** `exterior_tracer` restringido igual al actual en las columnas usadas; el modelo anidado igual bit a bit con y sin el cambio.
- **Del modelo:** la suite completa, las huellas y el restart.
- **Medición:** un script `validation/bayiw_cut/profile_step.py` que imprime s/paso, el escalado de 1 a 5 hilos y el reparto por pieza, para cada caso de la tabla del §3. Se corre al inicio y al final y se anota en el README del corte.

## 6. Fuera de alcance

- MPI y la descomposición del dominio.
- El backend JAX.
- Cambiar la física o los esquemas: solo cambia dónde y cómo se calcula.
