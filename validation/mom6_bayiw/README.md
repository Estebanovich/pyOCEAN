# pyOCEAN contra MOM6 en BayIW

- **Spec:** `docs/superpowers/specs/2026-10-13-mom6-bayiw-benchmark-design.md`
- **Plan:** `docs/superpowers/plans/2026-10-13-mom6-bayiw-benchmark.md`

El costo de HYCOM pasó al benchmark de los cuatro modelos (decisión de Esteban, 2026-10-13).

## Configuración de MOM6

**Build:** MOM6 + FMS de `~/pyOCEAN_refs`, `.testing/build/repro` (`-g -O2`, `FC=mpif90`; `run_mom6.sh build`, 70 s). FMS se compiló en `-O0`. 5 procesos MPI (`LAYOUT = 1, 5`).

**Entradas** (`inputs.py`), desde `bayiw`:
- supermalla mosaico de la malla estirada;
- profundidad efectiva de MITgcm;
- T en niveles z, con centros cuyos puntos medios son las interfaces de MITgcm (MOM6 reconstruye así los bordes de origen), de modo que el salto de dos capas entra nítido; S constante;
- viento en la cara norte de cada fila;
- objetivos de densidad de `bayiw`.

**Física** (`config.py`):
- EOS lineal (ρ = 999.8 − 0.19996 (T − T_ref), confirmado en el registro de MOM6);
- g = 9.81;
- Laplaciano de 100 m² s⁻¹, vertical 1e-5, plano f, energía cinética Arakawa, sin ley de fricción de fondo;
- remapeo PPM_H4 y estado inicial remapeado desde los niveles z;
- Flather + Orlanski hacia el reposo, con T y S del borde tomadas del interior al inicio;
- `ENABLE_BUGS_BY_DEFAULT = False`, salvo esas tres opciones de OBC;
- Δt = 120 s con barotrópico *split-explicit*.

**Diferencias que impone MOM6/FMS o que son esquemas propios de cada modelo (clase c):**
- **La sección tiene 10 columnas** (tierra | la columna | 8 de tierra). FMS compara todos los halos, también el sur, el norte, el de la supermalla y el barotrópico, con el tamaño en x (`mpp_domains_define.inc:899-906`). Las columnas de tierra no cambian la física.
- **El viento de archivo es constante por registro de 1200 s, centrado en los registros de MITgcm**, que interpola entre ellos.
- **La primera salida de MOM6 es a 0.25 h** (no hay t = 0). Las anomalías se miden desde ahí; el efecto es ≤ 0.01 en los cocientes.
- **Valores por defecto de MOM6 que no se cambiaron**, porque son esquemas propios:
  - advección de trazadores PLM (MITgcm usa segundo orden; pyOCEAN, WENO5);
  - `KHTR` a lo largo de las capas del modelo (isopícnico en HYCOM1);
  - el fondo sigue siendo viscoso sin ley de arrastre: `a_cpl = Kv/(½h)` (`MOM_vert_friction.F90:2271`);
  - `OBC_RADIATION_MAX = 1` (CMAX 0.45 en MITgcm y pyOCEAN);
  - `BOUND_CORIOLIS = False`.
- **HYBGEN (el algoritmo de HYCOM) no corre en este build.** Se detiene con "open_file: FMS I/O requires a domain input" al escribir su archivo de coordenada (`MOM.F90:3613` → `write_Hybgen_coord_file`, `MOM_hybgen_regrid.F90:260`), también con 1 proceso. No se modifica MOM6; la comparación de pyOCEAN con hybgen sigue siendo la de columna (`validation/mom6_column/`).

## Resultados en la sección 2-D (60 h, contra MITgcm `orl`)

`python validation/mom6_bayiw/compare.py 2layer|real`; las tablas por modelo están en `compare_<caso>_<coord>.txt` (`bayiw_2d/compare.py <caso> mom6<coord> orl`). Costo de la sección: 60 h en 22 s (HYCOM1) y 44 s (z\*) con 5 procesos.

### Dos capas
```
# BayIW 2-D section 2layer with bay: rms model/MITgcm · zero-lag corr (MITgcm 'orl' = reference)

## V110
station            pyOCEAN z*     pyOCEAN híbrida             MOM6 z*         MOM6 HYCOM1
Gamma            0.97 ·  0.98        0.95 ·  0.92        1.00 ·  1.00        0.96 ·  0.98
C                0.78 ·  0.97        0.78 ·  0.82        0.93 ·  1.00        0.80 ·  0.97
Omega            2.19 ·  0.97        1.44 ·  0.94        1.98 ·  0.98        1.14 ·  1.00
Seiche2          1.77 · -0.36        0.71 ·  0.86        1.21 · -0.19        0.80 ·  0.46
Seiche           1.43 ·  0.97        2.10 ·  0.91        1.63 ·  0.96       13.89 · -0.70

## T110
station            pyOCEAN z*     pyOCEAN híbrida             MOM6 z*         MOM6 HYCOM1
Gamma            1.01 ·  1.00        0.07 ·  1.00        1.02 ·  1.00        0.11 ·  0.98
C                1.30 ·  1.00        0.78 ·  0.99        1.30 ·  1.00        0.34 ·  1.00
Omega            2.26 ·  0.91        0.07 ·  0.92        2.30 ·  0.91        0.37 ·  0.82
Seiche2          1.59 ·  0.97        0.07 ·  0.96        1.56 ·  0.97        0.30 ·  0.44
Seiche           2.33 ·  0.92        0.16 ·  0.92        2.29 ·  0.92        2.13 ·  0.68

## zeta
station            pyOCEAN z*     pyOCEAN híbrida             MOM6 z*         MOM6 HYCOM1
Gamma            0.90 ·  1.00        0.07 · -0.00        0.98 ·  1.00        0.66 ·  0.99
C                1.01 ·  1.00        1.17 ·  0.98        1.00 ·  1.00        1.02 ·  1.00
Omega            1.12 ·  0.97        1.65 ·  0.81        1.10 ·  0.97        1.15 ·  0.98
Seiche2          0.99 ·  0.99        1.54 ·  0.96        0.99 ·  0.99        1.08 ·  0.99
Seiche           1.00 ·  1.00        1.37 ·  0.99        0.97 ·  0.99        1.06 ·  0.77

## T section at 24 h: relative rms error of the anomaly; interface thickness [m] (MITgcm 8.1)
pyOCEAN z*          0.43, interface 8.2
pyOCEAN híbrida     0.72, interface 8.0
MOM6 z*             0.44, interface 8.2
MOM6 HYCOM1         0.91, interface 28.3
```

### Realista (febrero)
```
# BayIW 2-D section real with bay: rms model/MITgcm · zero-lag corr (MITgcm 'orl' = reference)

## V110
station            pyOCEAN z*     pyOCEAN híbrida             MOM6 z*         MOM6 HYCOM1
Gamma            1.01 ·  1.00        1.02 ·  1.00        1.01 ·  1.00        1.01 ·  1.00
C                1.00 ·  1.00        1.01 ·  1.00        1.00 ·  1.00        0.99 ·  1.00
Omega            1.00 ·  1.00        0.99 ·  1.00        0.99 ·  1.00        1.14 ·  0.97
Seiche2          1.00 ·  1.00        0.99 ·  1.00        0.95 ·  0.98        2.60 ·  0.17
Seiche           1.09 ·  0.99        1.21 ·  0.98        0.93 ·  0.99        1.65 ·  0.65

## T110
station            pyOCEAN z*     pyOCEAN híbrida             MOM6 z*         MOM6 HYCOM1
Gamma            1.00 ·  1.00        0.86 ·  1.00        1.01 ·  1.00        0.86 ·  1.00
C                1.01 ·  1.00        0.96 ·  1.00        1.01 ·  1.00        0.95 ·  1.00
Omega            1.00 ·  1.00        0.98 ·  1.00        1.05 ·  0.99        1.48 ·  0.94
Seiche2          0.99 ·  1.00        1.06 ·  0.99        0.92 ·  0.99        3.21 ·  0.36
Seiche           1.03 ·  0.99        1.04 ·  0.95        1.00 ·  1.00        1.51 ·  0.22

## T section at 24 h: relative rms error of the anomaly
pyOCEAN z*          0.04
pyOCEAN híbrida     6.71
MOM6 z*             0.27
MOM6 HYCOM1        14.50
```

## Costo en el corte BayIW (198 × 159, dos capas con bahía, viento, fronteras radiantes; 5 procesos o hilos)

MOM6: `run_mom6.sh cost <coord> <dt> <horas>` (bucle principal según los relojes de FMS; incluye escribir la salida cada 900 s). pyOCEAN: 30 pasos cronometrados tras 5 de calentamiento, sin salida.

| s por hora simulada | Δt = 120 s | Δt = 240 s | Δt = 360 s | Δt = 600 s |
|---|---|---|---|---|
| MOM6 z\* (50 capas) | 8.97 | 4.51 | 3.06 | 1.94 (6 h) |
| pyOCEAN z\* (50 capas) | 9.13 | — | — | 1.82 (5.8 h) |
| MOM6 HYCOM1 (25 capas) | 4.67 | 2.37 | 1.62 | 1.02 (6 h) |
| pyOCEAN híbrida (25 capas) | 4.95 | — | — | 0.99 (5.8 h) |
| MITgcm z (50 niveles, Δt 30 s de Esteban) | 4.4 | | | |

- **A igual Δt, pyOCEAN cuesta lo mismo que MOM6, a ±10 %.** MOM6 incluye la salida y su `LAYOUT = 1×5` desbalancea el ALE entre procesos; pyOCEAN no escribe. La diferencia que veíamos con MITgcm viene de que el paso ALE remapea y MITgcm no.
- **Los dos son estables ~6 h con Δt = 600 s** (pyOCEAN sin truncamientos; MOM6 gracias a su *split-explicit*, pyOCEAN a su superficie libre implícita). **No se ha evaluado la precisión con Δt > 120 s.**
- **Reparto de MOM6 en el corte** (Δt 120 s, fracción del bucle principal; z\* / HYCOM1):
  - dinámica 63 / 59 %;
  - corrección de la continuidad 18.6 / 17.0 %;
  - ALE (regrid y remapeo) 18.6 / 22.8 %;
  - trazadores 9.8 / 9.0 %, de ellos advección 4.8 / 4.5 %;
  - precálculos barotrópicos 4.6 %;
  - diagnósticos 2.3 %.
- **Frente a pyOCEAN z\*** (`profile_final.txt`):
  - el remapeo de pyOCEAN (~13.5 %) cuesta menos que el ALE de MOM6 (18.6 %);
  - la continuidad PPM de pyOCEAN (~7 %) cuesta menos que la corrección de la continuidad de MOM6;
  - la advección de T de pyOCEAN, WENO5 con tres etapas SSP-RK3 (~11 %), cuesta más del doble que la PLM de una etapa de MOM6 (4.8 %).

## Clasificación

| Diferencia | Clase | Razonamiento |
|---|---|---|
| z\* (pyOCEAN y MOM6) frente a MITgcm: T110 inflada 1.3–2.3× y V110 de Seiche2 anticorrelacionada | (c) | pyOCEAN z\* y MOM6 z\* dan lo mismo en todas las métricas; es la coordenada z\* frente al z-level de MITgcm |
| MOM6 HYCOM1: interfaz de 28 m, V110 de Seiche 13.9× (corr. −0.70), realista peor en la bahía | (c) del modelo de referencia | P1M con bordes promediados sin preservar las interfaces isopícnicas: el mismo mecanismo que pyOCEAN corrigió con `keep_isopycnal`. No es un artefacto del muestreo (con PCM da 13.88) |
| pyOCEAN híbrida en dos capas: ζ 1.17–1.65 (MOM6 HYCOM1 1.02–1.15) | **(a) candidato** | MOM6 se parece más a MITgcm aquí, aunque difunde la interfaz, lo que amortigua su movimiento. Diferido al plan de la híbrida (decisión de Esteban) |
| Error de la sección T a 24 h en el caso realista: pyOCEAN híbrida 6.71, MOM6 HYCOM1 14.5 (z\* 0.04/0.27) | (c), abierto | En las dos híbridas es un error local de 0.25 / 0.57 °C cerca del nivel 33 en las filas 334–335 (cabecera somera de la bahía), frente a una anomalía de MITgcm de 1.8e-3 °C. Muestreo de capas híbridas cerca del fondo; va al plan de la híbrida |
| Esquemas propios: advección PLM / WENO5 / segundo orden, `KHTR` por capas, fondo viscoso, `OBC_RADIATION_MAX` | (c) | Elección numérica de cada modelo |

## Qué debe mejorar pyOCEAN

1. **ζ de la híbrida en dos capas** (17–65 % de más): diagnóstico en el plan de la híbrida.
2. **El muestreo y las capas casi vacías de la híbrida cerca del fondo** en la bahía somera (error de sección 6.7 en el caso realista).
3. **El costo de la advección de T:** WENO5 + SSP-RK3 cuesta más del doble que la PLM de MOM6. Conviene una opción PPM/PLM de una etapa para corridas largas, evaluada con la mezcla espuria.
4. **La precisión con Δt grande:** pyOCEAN es estable con 600 s. Hay que evaluar cuánto se pierde contra 120 s (en ondas internas y seiche) antes de recomendarlo.
