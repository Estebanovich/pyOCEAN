# Benchmark baroclínico de BayIW contra MITgcm

**Spec:** `docs/superpowers/specs/2026-10-12-bayiw-baroclinic-benchmark-design.md`.
**Herramientas:**
- `bayiw.py`: entradas de MITgcm y modelo ALE con sus parámetros;
- `run_pyocean.py`: corredor, salidas cada 900 s;
- `compare.py`: métricas del spec §6.

**Estado (2026-10-12):** fase 0 hecha (montaje, corredor, comparación y fronteras con rotación). Esteban redefinió el orden por costo:
1. sección 2-D del eje de la bahía contra la misma sección en MITgcm;
2. `internal_wave` de MITgcm, que necesita forzamiento baroclínico en la frontera abierta;
3. BayIW 3-D recortada contra MITgcm recortado;
4. BayIW completa, opcional.

Antes se optimizó el paso ALE (abajo).

## Rendimiento del paso ALE

Plan: `docs/superpowers/plans/2026-10-12-ale-step-performance.md`.
- **Máquina:** M3 Pro, 5 P-cores (`NUMBA_NUM_THREADS=5`).
- **Medición:** `scripts/profile_ale_step.py`. BayIW es dos capas con bahía, 560 × 352, Δt = 120 s, fronteras abiertas con esponja de 30 celdas.

**Segundos por paso tras cada tarea:**

| Paso | BayIW z\* (50 capas) | BayIW híbrida (25 capas) | Caja 200 × 120 × 30 |
|---|---|---|---|
| Línea base (NumPy, superficie libre directa) | 8.48 | 4.77 | 0.59 |
| Superficie libre CG + LU desfasado, con conservación exacta (MITgcm `exactConserv`) | 7.75 | | |
| Grosor de cara: núcleo Numba + caché por identidad | 7.15 | | |
| Transporte de grosor PPM: núcleos (bordes, volumen barrido, Newton global, residuo HYCOM) | 4.94 | 2.41 | |
| Advección: caras sin flujo | 4.86 | | |
| Laplaciano 3-D con espaciamientos en caché | 4.58 | | |
| Coriolis PV, vorticidad y energía cinética: núcleos | 4.03 | | |
| Fracciones de cara, separación baroclínica y capas delgadas: núcleos | 3.46 | | |
| Advección sin paso vertical en ALE | 3.40 | | |
| Difusión horizontal: núcleo; alturas de capa en caché | **3.32** | **1.58** | **0.19** |
| Lo mismo, sin el diagnóstico de mezcla espuria (`mix_diagnostic=False`) | **2.72** | **1.44** | |

**Aceleración:**
- z\*: 2.6× con el diagnóstico de mezcla, 3.1× sin él;
- híbrida: 3.0× y 3.3×;
- caja: 3.1×.

**Resultados sin cambios:**
- cada núcleo coincide bit a bit con su versión NumPy, que queda como oráculo en los tests (PPM: hasta 1e-12, por el orden de las sumas);
- σ es idéntico bit a bit;
- ALE coincide a 1e-10 tras 100 pasos en el lock exchange, el canal, el escalón y la caja 2-D;
- la superficie libre CG difiere de la directa ≤ 1e-9 m y conserva volumen y calor al redondeo.

**Frente a MITgcm** (0.29 s/paso con 8 procesos MPI, Δt = 30 s → 35 s por hora simulada):
- pyOCEAN z\* con Δt = 120 s: 82 s por hora simulada (2.3× más lento; antes, 7×);
- pyOCEAN híbrida con 25 capas: 43 s por hora (1.2×).

**Perfil final** (z\*, sin diagnóstico, 2.65 s/paso bajo cProfile):

| Pieza | Parte del paso |
|---|---|
| Advección | 15 % |
| Gradiente de presión FV | 13 % |
| Remapeo | 9 % |
| Copias `ascontiguousarray` | 8 % |
| Superficie libre | 7.5 % |
| PPM | 7.5 % |
| `cumsum` | 7 % |
| Resto | < 5 % cada una |

Ninguna pieza ahorraría ya más del 5 % por sí sola (regla del spec), y la meta de 1.5 s/paso solo se alcanza con la híbrida. Para seguir bajando en z\* haría falta:
- fusionar el paso en menos núcleos, para evitar copias y arreglos temporales de 80–160 MB;
- o un backend JAX/GPU (etapa D del roadmap), o MPI.
