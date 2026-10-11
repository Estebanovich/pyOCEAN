# Rendimiento: cómo lo hacen otros modelos y qué hace pyOCEAN

Revisión del 2026-10-11, motivada por BayIW: MITgcm tarda ~1–1.4 h de reloj por corrida de 120 h y pyOCEAN tardaba ~2 h.

**Fuentes:** biblioteca local (FESOM2-JAX, DJ4Earth/Oceananigans, Veros-JAX, ROMS), código fuente de MITgcm en `/Volumes/ESTEBAN_MIT/MITgcm` y una búsqueda en Consensus.

## 1. Dónde se va el tiempo en pyOCEAN (BayIW, 560 × 352 × 50, M3 Pro)

| Versión | s/paso | Nota |
|---|---|---|
| NumPy puro | 31 | 75 % en la reconstrucción WENO; mucho tiempo de sistema por temporales de ~160 MB |
| WENO5 en Numba, 1 hilo | 6.5 | |
| + 5 hilos solo en WENO | 5.0 | limitado por el ancho de banda de memoria: el resto es NumPy de un hilo |
| + advección de trazadores fusionada, Thomas compilado, CG con LU desfasado | 3.3 | |

Por celda y paso estamos a ~1.7× de MITgcm: ~500 contra ~290 ns por núcleo. Pero usamos Δt = 300 s contra 30 s, así que las horas-núcleo por corrida son ~2–3 contra ~11.

## 2. Qué hacen los demás

1. **No crear arreglos temporales; fusionar todo el paso.**
   - MITgcm recorre `DO bj, DO bi` (tiles) y `DO k` (niveles), y calcula todos los términos de un nivel sobre rebanadas 2-D del tile, que caben en caché (verificado en `model/src/dynamics.F`). ROMS hace lo mismo con *tiling*.
   - Veros (Python + JAX) iguala a su referencia Fortran en CPU porque XLA fusiona las operaciones de arreglos (Häfner et al. 2021).
   - FESOM2-JAX compila el paso completo como un solo programa (`jax.lax.scan`). Aun así, sus kernels en CPU son ~2.7× más lentos que Fortran (Koldunov et al. 2026).
   - En MPAS-Ocean en Julia, quitar las asignaciones implícitas de memoria dio 10–20× (Bishnu et al. 2023).
   - La fusión profunda de kernels en Fortran dio hasta 2.6× más velocidad y 2.3× menos memoria (Remy et al. 2026).
   - Los resolvedores cartesianos de este tipo están limitados por el ancho de banda de memoria, no por la aritmética (Briand et al. 2025).
2. **Modo barotrópico.**
   - ROMS, CROCO, MOM6, HYCOM y NEMO usan *split-explicit* (Shchepetkin & McWilliams 2005): el barotrópico se subcicla explícitamente en 2-D y no hay solver global.
   - MITgcm y Veros usan superficie libre implícita con CG. En Veros, cambiar a AMG (pyAMG) hizo ese paso hasta 10× más rápido que el CG de Fortran (Häfner et al. 2018).
   - Con muchos procesos, la escalabilidad la limitan las sincronizaciones globales por iteración (Kang et al. 2023).
3. **GPU.** Oceananigans (Julia), Veros-JAX y FESOM2-JAX: una GPU reemplaza a cientos de núcleos, con menor consumo de energía.
4. **Precisión simple.** Veros corre su configuración global de 0.1° en float32 en GPU. MITgcm y FESOM2 usan float64; FESOM2 lo exige para la conservación en corridas de décadas.

### 2b. Detalles de los PDF ya locales (2026-10-11)

- **Veros 2021:**
  - NumPy es 3–5× más lento que Fortran.
  - JAX en CPU es 1.6× *más rápido* que Fortran en un proceso, porque usa hilos internamente, y como mucho 1.4× más lento con MPI.
  - JAX rinde por completo con más de 10⁵ celdas.
  - Los solves tridiagonales no se expresan bien como operaciones de arreglos: Veros usa un kernel propio en C/CUDA vía *custom call* de JAX, la única parte no Python.
  - La superficie libre se resuelve con PETSc.
- **Veros 2018:** cambiar CG por AMG (pyAMG) hizo el solver barotrópico hasta 10× más rápido que el de Fortran.
- **Oceananigans 2023:**
  - Fusiona toda la tendencia de cada variable pronóstica en un solo kernel por celda (i, j, k), con solo dos campos intermedios (presión hidrostática y difusividad vertical).
  - Los modelos tradicionales reservan 10–100× la memoria mínima.
- **Bishnu 2023:** quitar asignaciones de memoria dio 10–20× en un solo núcleo.
- **Remy 2026:** la fusión profunda dio hasta 2.6× más velocidad y 2.3× menos memoria en NEMO y CROCO.
- **Briand 2025:**
  - Estos códigos están limitados por el ancho de banda de memoria.
  - El rendimiento cae cuando el dominio no cabe en caché.
  - Cada kernel debe ser una sección paralela de bajo costo.
  - Más de un hilo por núcleo no ayuda.
- **Kang 2023:** con muchos procesos, el solver implícito lo limitan las sincronizaciones globales; con 5 núcleos no es problema.

## 3. Qué significa para pyOCEAN

| Opción | Ganancia esperada | Costo | Encaje con la hoja de ruta |
|---|---|---|---|
| A. Seguir fusionando en Numba: tendencias de momento en un kernel, la etapa SSP-RK3 dentro del kernel de advección, buffers preasignados | ~2× más sobre 3.3 s/paso | medio; código de bucles distinto del estilo NumPy | ninguno para la etapa D: no es diferenciable |
| B. Backend JAX (etapa D adelantada): el código ya es estilo NumPy, `jax.jit` fusiona solo, se diferencia con `jax.grad`, y la misma fuente correría en una GPU NVIDIA | Veros: igual a Fortran en CPU; FESOM2: 2.7× más lento por kernel | alto: solver de η en JAX (CG sin matriz con `custom_linear_solve`, como FESOM2), Thomas con `lax.scan`, sin escrituras in situ | sí: es la etapa D |
| C. Split-explicit barotrópico (etapa C) | elimina el solver global; paraleliza mejor | alto; cambia la numérica, hay que revalidar BayIW, mareas y mojado-secado | sí: etapa C |
| D. pyAMG para la superficie libre | menor: el solver ya es ~0.2 s/paso con LU desfasado | bajo (requiere instalar pyAMG) | — |

**Decisión informada.** Lo que acelera es fusionar y no crear temporales (Oceananigans, Bishnu, Remy). Eso se logra igual con bucles Numba al estilo Oceananigans o con XLA en JAX al estilo Veros. Solo JAX da además gradientes (etapa D) y portabilidad a GPU NVIDIA; en esta Mac, JAX solo usaría CPU. Veros muestra el camino híbrido: JAX para el código de arreglos, un kernel propio para el tridiagonal y un solver externo para η.

**Recomendación:**
1. Terminar lo empezado en Numba (el kernel de momento ya está escrito y su test está pendiente).
2. Invertir el resto del esfuerzo de rendimiento en un prototipo JAX del paso σ estratificado, midiendo en esta Mac antes de comprometerse. JAX resolvería a la vez la fusión, la diferenciabilidad y el uso futuro de GPU. Instalar JAX requiere el visto bueno de Esteban.

## 4. Medición del prototipo JAX (2026-10-11)

`prototypes/bench_jax_vs_numba.py`: lado derecho de la advección de trazadores (2 trazadores × 50 × 352 × 560). JAX y Numba coinciden con NumPy (1e-13 y 0).

| Versión | Reloj | Núcleos ocupados | CPU por llamada |
|---|---|---|---|
| Numba fusionado, 1 hilo | 0.674 s | 1 | 0.67 s |
| Numba fusionado, 5 hilos | 0.177 s | 5 | ~0.9 s |
| JAX/XLA (jit) | 0.304 s | **~7.7** | ~2.4 s |

- **XLA no se puede limitar a un hilo en esta Mac** con jaxlib 0.4.35: ignora `--xla_cpu_multi_thread_eigen=false` e `intra_op_parallelism_threads=1`, también con el runtime anterior. Usa también los núcleos de eficiencia, así que no sirve para corridas aquí sin violar la regla de ≤ 5 núcleos P.
- **Por CPU, Numba fusionado gana ~3.6×:** 0.67 contra ~2.4 s de CPU, aunque parte de los núcleos de XLA son E-cores, más lentos. Con 5 hilos, Numba también gana en tiempo de reloj (0.18 contra 0.30 s) usando menos núcleos.
- Es coherente con FESOM2-JAX (kernels ~2.7× más lentos que Fortran en CPU).

**Decisión de backend (paso 2 del plan):**
- **En CPU, Numba fusionado** sigue siendo el motor de producción.
- **JAX** solo se justifica por gradientes (etapa D) y GPU NVIDIA, y se prueba fuera de esta Mac o con una versión de jaxlib que respete el límite de hilos.
- Mantener los kernels como funciones puras sobre arreglos facilita portarlos después.

## Referencias

- Häfner, D., Jacobsen, R.L., Eden, C. et al. (2018). Veros v0.1. GMD 11, 3299. [10.5194/gmd-11-3299-2018](https://doi.org/10.5194/gmd-11-3299-2018)
- Häfner, D., Nuterman, R. & Jochum, M. (2021). Fast, cheap, and turbulent: global ocean modeling with GPU acceleration in Python. JAMES. [10.1029/2021MS002717](https://doi.org/10.1029/2021ms002717)
- Koldunov, N. et al. (2026). FESOM2-JAX v1.0. arXiv. [10.48550/arxiv.2608.01546](https://doi.org/10.48550/arxiv.2608.01546) (local)
- Bishnu, S. et al. (2023). Julia vs Fortran-MPI, MPAS-Ocean. GMD 16, 5539. [10.5194/gmd-16-5539-2023](https://doi.org/10.5194/gmd-16-5539-2023)
- Remy, J. et al. (2026). Poseidon: deep kernel fusion for ocean models. IJHPCA. [10.1177/10943420261456347](https://doi.org/10.1177/10943420261456347)
- Briand, L. et al. (2025). Julia vs C++ Kokkos for Cartesian CFD. IJHPCA. [10.1177/10943420251341179](https://doi.org/10.1177/10943420251341179)
- Shchepetkin, A.F. & McWilliams, J.C. (2005). ROMS split-explicit kernel. Ocean Modell. 9, 347. (local)
- Kang, H.-G. et al. (2023). Implicit barotropic solver for MPAS-O. IJHPCA. [10.1177/10943420231205601](https://doi.org/10.1177/10943420231205601)
- Moses, W.S. et al. (2026). DJ4Earth. JAMES. [10.1029/2025MS005615](https://doi.org/10.1029/2025ms005615) (local)
