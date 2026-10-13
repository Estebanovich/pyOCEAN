# pyOCEAN contra MOM6 (y el costo de HYCOM) en BayIW

- **Spec:** `docs/superpowers/specs/2026-10-13-mom6-bayiw-benchmark-design.md`
- **Plan:** `docs/superpowers/plans/2026-10-13-mom6-bayiw-benchmark.md`

## Configuración de MOM6

**Build:** MOM6 + FMS de `~/pyOCEAN_refs`, `.testing/build/repro` (`-g -O2`, `FC=mpif90`; `run_mom6.sh build`, 70 s). FMS se compiló en `-O0`. 5 procesos MPI (`LAYOUT = 1, 5`).

**Entradas** (`inputs.py`), desde `bayiw`:
- supermalla mosaico de la malla estirada;
- profundidad efectiva de MITgcm;
- T en niveles z, con centros cuyos puntos medios son las interfaces de MITgcm (MOM6 reconstruye así los bordes de origen), de modo que el salto de dos capas entra nítido; S constante;
- viento en la cara norte de cada fila;
- objetivos de densidad de `bayiw`.

**Física** (`config.py`), la de las corridas de Esteban:
- EOS lineal, Laplaciano de 100 m² s⁻¹, vertical 1e-5, sin fricción de fondo, deslizamiento libre, plano f, energía cinética Arakawa;
- remapeo PPM_H4 y estado inicial remapeado desde los niveles z (`Z_INIT_ALE_REMAPPING`);
- fronteras Flather + Orlanski hacia el reposo;
- Δt = 120 s con barotrópico *split-explicit* (`DTBT = −0.98`).

**Diferencias que impone MOM6 o FMS:**
- **La sección tiene 10 columnas** (tierra | la columna | 8 de tierra), no las 3 de MITgcm. FMS compara todos los halos (también el sur, el norte, el de la supermalla y el barotrópico) con el tamaño en x (`mpp_domains_define.inc:899-906`). Las columnas de tierra no cambian la física.
- **El viento de archivo es constante por registro de 1200 s**; MITgcm interpola entre registros (desfase máximo de medio registro).
- **HYBGEN (el algoritmo de HYCOM) no corre en este build.** Toda corrida se detiene con "open_file: FMS I/O requires a domain input" al escribir su archivo de coordenada (`MOM.F90:3613` → `write_Hybgen_coord_file`, `MOM_hybgen_regrid.F90:260`), también con 1 proceso. No se modifica MOM6. La comparación de pyOCEAN con hybgen sigue siendo la de columna (`validation/mom6_column/`).

**Costo de la sección:** 60 h en 22 s (HYCOM1, 25 capas) y 44 s (z\*, 50 capas) con 5 procesos.

## Resultados en la sección 2-D (60 h, contra MITgcm `orl`)

`python validation/mom6_bayiw/compare.py 2layer|real`; las tablas por modelo están en `compare_<caso>_<coord>.txt` (`bayiw_2d/compare.py <caso> mom6<coord> orl`).

### Dos capas
```
# BayIW 2-D section 2layer with bay: rms model/MITgcm · zero-lag corr (MITgcm 'orl' = reference)

## V110
station            pyOCEAN z*     pyOCEAN híbrida             MOM6 z*         MOM6 HYCOM1
Gamma            0.97 ·  0.98        0.95 ·  0.92        1.00 ·  1.00        0.96 ·  0.97
C                0.78 ·  0.97        0.78 ·  0.82        0.93 ·  1.00        0.80 ·  0.97
Omega            2.19 ·  0.97        1.44 ·  0.94        1.98 ·  0.98        1.14 ·  1.00
Seiche2          1.77 · -0.36        0.71 ·  0.86        1.22 · -0.20        0.80 ·  0.46
Seiche           1.43 ·  0.97        2.10 ·  0.91        1.63 ·  0.96       13.88 · -0.70

## T110
station            pyOCEAN z*     pyOCEAN híbrida             MOM6 z*         MOM6 HYCOM1
Gamma            1.01 ·  1.00        0.07 ·  1.00        1.02 ·  1.00        0.11 ·  0.98
C                1.30 ·  1.00        0.78 ·  0.99        1.30 ·  1.00        0.34 ·  1.00
Omega            2.26 ·  0.91        0.07 ·  0.92        2.30 ·  0.91        0.37 ·  0.82
Seiche2          1.59 ·  0.97        0.07 ·  0.96        1.57 ·  0.97        0.30 ·  0.44
Seiche           2.33 ·  0.92        0.16 ·  0.92        2.29 ·  0.92        2.13 ·  0.68

## zeta
station            pyOCEAN z*     pyOCEAN híbrida             MOM6 z*         MOM6 HYCOM1
Gamma            0.90 ·  1.00        0.07 · -0.00        0.97 ·  1.00        0.66 ·  0.99
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
Seiche2          1.00 ·  1.00        0.99 ·  1.00        0.95 ·  0.98        2.61 ·  0.17
Seiche           1.09 ·  0.99        1.21 ·  0.98        0.93 ·  0.99        1.65 ·  0.65

## T110
station            pyOCEAN z*     pyOCEAN híbrida             MOM6 z*         MOM6 HYCOM1
Gamma            1.00 ·  1.00        0.86 ·  1.00        1.01 ·  1.00        0.85 ·  1.00
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
