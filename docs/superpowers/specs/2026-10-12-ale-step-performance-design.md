# Rendimiento del paso ALE (núcleos Numba guiados por el perfil)

**Fecha:** 2026-10-12. **Estado:** diseño aprobado por Esteban en la conversación (enfoque A); falta revisar el texto escrito.
**Rama:** `ale-plan-b`.
**Motivo:** el benchmark de BayIW (spec `2026-10-12-bayiw-baroclinic-benchmark-design.md`) cuesta 8.6 s/paso en 560 × 352 × 50 (z\*, PPM, fronteras abiertas, 5 P-cores) contra 0.29 s/paso de MITgcm con 8 procesos MPI. Esteban pidió "la mejor optimización posible" antes de seguir con las simulaciones.

## 1. Diagnóstico (perfil de 5 pasos de BayIW dos capas con bahía, z\*, Δt = 120 s)

| Pieza | s/paso | % | Causa |
|---|---|---|---|
| Transporte de grosor PPM (`ppm_transport_x`: `_swept_thickness`, `ppm_edges_x`, `_ppm_fluxes`) | 2.4 | 33 | NumPy en un hilo; cada iteración de Newton recorre arreglos 3-D completos |
| Superficie libre (`spsolve`, SuperLU) | 1.0 | 14 | refactorización directa cada paso, en un hilo |
| Advección de trazadores (WENO5, ya en Numba) | 1.0 | 14 | — |
| `face_transport_thickness` + `cumsum` | 0.8 | 11 | se recalcula 6 veces por paso con los mismos datos |
| Resto: `coriolis_pv`, `_laplacian`, `fv_pgf`, `zero_thin_bed_velocity`, `baroclinic_split`, `face_fractions`, `_diffuse_h`, `kinetic_energy`, `truncate_cfl` | ≈2 | 28 | muchas operaciones NumPy pequeñas con temporales de ≈80 MB |

Total bajo el perfilador: 7.1 s/paso (8.6 s sin perfilador).

## 2. Enfoque A (elegido)

- **Núcleos guiados por el perfil:** cada pieza caliente se reescribe como un núcleo `numba.njit(parallel=True, cache=True)`:
  - `prange` sobre filas o columnas;
  - recorridos fusionados y sin temporales de dominio completo;
  - nada se calcula dos veces dentro del paso.
- **La versión NumPy es el oráculo:** la versión actual de cada función queda como implementación de referencia en los tests (`*_numpy`, como ya hace `_flux_form_u_numpy`), y el núcleo nuevo debe coincidir con ella.
- **Orden y criterio de parada:**
  - se mide, se optimiza la pieza más cara y se repite;
  - se para cuando ninguna pieza pase del 10 % del paso o cuando la siguiente ganancia sea menor del 5 % del total.
- **Descartados:**
  - un solo núcleo para todo el paso: difícil de probar y de mantener;
  - JAX ahora: es la etapa D, y en CPU no supera a Numba.

## 3. Piezas

1. **Superficie libre:**
   - en ALE, `solver="cg"` con `precond="lu"` por defecto (gradiente conjugado con la LU de una matriz desfasada, ya en `core/dynamics/wetting_drying.py`, 1–3 iteraciones en el Alto Golfo);
   - el modo σ conserva su valor por defecto (`direct`), así sigue idéntico bit a bit;
   - **verificación:** η con CG y η con la solución directa coinciden con `cg_rtol` = 1e-10 (diferencia ≤ 1e-9 m en BayIW tras 10 pasos).
   - Si la LU desfasada sigue pesando, se evalúa reusarla más pasos (`refactor_iterations`).
2. **Transporte de grosor PPM** (`ppm_transport_x`): un núcleo por cara-columna que hace:
   - bordes PPM con el limitador positivo;
   - volumen barrido;
   - Newton con bisección, local a cada columna (como MOM6 `zonal_flux_adjust`, que también itera por fila);
   - reparto del residuo (HYCOM).
   - **Verificación:** igual a la versión NumPy hasta 1e-12 relativo en casos aleatorios, con capas desvanecidas, recorte al fondo, tierra y bordes del dominio.
3. **Grosor de cara para transporte** (`face_transport_thickness`):
   - un núcleo;
   - en `_step_ale` se calcula una vez para hⁿ y una para h_new, y se reutiliza en el grosor medio, las fracciones, la viscosidad y la regla de capas delgadas.
   - **Verificación:** idéntico a NumPy (mismas operaciones, ≤ 1e-14).
4. **El resto, mientras el perfil lo justifique:**
   - Coriolis PV y energía cinética (`vector_invariant`);
   - Laplaciano horizontal;
   - `baroclinic_split` y `face_fractions`;
   - gradiente de presión FV;
   - regla de capas delgadas;
   - difusión horizontal;
   - truncamiento CFL.

   Cada uno con su oráculo NumPy.

## 4. Reglas

- **Hilos:** ≤ 5 (`NUMBA_NUM_THREADS=5`; cada núcleo llama `numba.set_num_threads(_THREADS[0])` como el resto del modelo). Nunca los núcleos de eficiencia.
- **Precisión:** float64; sin `fastmath` (cambiaría el orden de las sumas y rompería la comparación con el oráculo).
- **Modo σ:** no se toca; sigue idéntico bit a bit (`np.array_equal` en una corrida σ con fronteras abiertas, viento y f ≠ 0).
- **Modo ALE:** la regresión del modelo completo coincide con la versión actual a nivel de redondeo:
  - lock exchange híbrido;
  - canal de onda interna con fronteras;
  - seamount z\*;
  - escalón de la boca;
  - BayIW 1 h.

  Diferencias ≤ 1e-10 relativo en u, T y h tras 100 pasos. Con CG en la superficie libre: ≤ la tolerancia del solver propagada, documentada.
- **Sin dependencias nuevas.**

## 5. Pruebas

- **Por núcleo:** núcleo contra oráculo NumPy (casos aleatorios + capas desvanecidas + tierra + bordes + caras abiertas).
- **Del modelo:**
  - la suite completa pasa;
  - las comparaciones bit a bit (σ) y a redondeo (ALE) del §4.
- **Rendimiento:** `scripts/profile_ale_step.py` (nuevo, versionado) mide s/paso y el perfil por pieza en BayIW dos capas con bahía, z\* e híbrida, y en una malla pequeña para CI rápido. Los resultados van a una tabla en el README del benchmark y en `CLAUDE.md`.

## 6. Meta y criterio de éxito

- **Meta:** BayIW 560 × 352 × 50 z\* en ≤ 1.5 s/paso (≥ 5× más rápido), sin cambiar los resultados más allá del redondeo.
- **Si no se alcanza:** se reporta el perfil final, qué domina y qué haría falta para seguir bajando (JAX/GPU en la etapa D, o MPI).

## 7. Fuera de alcance

- JAX y GPU (etapa D);
- MPI o descomposición del dominio;
- float32;
- cambios de física o de esquemas numéricos: el resultado debe ser el mismo.
