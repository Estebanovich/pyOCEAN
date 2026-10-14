# HYCOM: compuerta de factibilidad (spec 2026-10-13-ilicak-four-model-benchmark-design §A3)

**Resultado (2026-10-13): el build pasa; el caso de costo queda pendiente de un anexo.**

```bash
bash validation/ilicak_benchmark/hycom/gate.sh
```

- **Build:** HYCOM 2.3 (`~/pyOCEAN/HYCOM-src`, sin modificar) en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/hycom/build`, ~35 s.
- **Cambios, solo en la copia de `config/generic-gnu-relo_mpi`:** se quitan `-mavx` y `-m64`, que son banderas de x86 que gfortran en arm64 no acepta.
- **EOS:** HYCOM la elige al compilar. Sin `CPP_EXTRAS="-DEOS_SIG0 -DEOS_7T"` falla en `mod_incupd.F90` (`sig`, `sigver` sin definir).
- **Cuenca en reposo (10 pasos, 5 procesos MPI): no se corrió.** `HYCOM-src` no trae entradas de ejemplo, y HYCOM necesita:
  - `regional.grid` y `regional.depth` en `.a/.b`;
  - `blkdat.input`, `patch.input`;
  - archivos de relajación y de forzamiento.

  Escribirlas requiere el escritor `.a/.b` que el plan de la parte A no especifica. Como dice el plan, aquí se detiene el trabajo con HYCOM. El caso de costo del overflow (100 capas híbridas, EOS de 7 términos con S constante) va en un anexo corto que Esteban aprueba.
