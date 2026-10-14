# HYCOM: costo en el overflow de Ilicak

- **Spec:** `docs/superpowers/specs/2026-10-13-ilicak-four-model-benchmark-design.md` §A3
- **Plan:** `docs/superpowers/plans/2026-10-13-ilicak-benchmark-hycom-addendum.md`

**Resultado (2026-10-14):** HYCOM 2.3 corre el overflow con ν = 1000. Su Δt máximo estable es 20 s, con 5.41 s por hora simulada en 1 proceso y 1.43 s en 5. Es un aprobado marginal: si no se acepta ningún grosor negativo, el Δt máximo es 10 s, con 10.9 s/h. Los tiempos se repitieron el 2026-10-14 con la máquina libre: la primera medición (5.75 y 2.84 s/h) la contaminaron tres procesos MPI huérfanos de MOM6 que llevaban 15 h al 100 %. Solo se mide el costo: la EOS, la mezcla y los bordes de HYCOM no son los de la física común, así que sus resultados no entran en las tablas de física.

```bash
bash validation/ilicak_benchmark/hycom/gate.sh                       # build (~35 s)
validation/ilicak_benchmark/hycom/run_hycom.sh <npes> <baclin> <batrop> <horas> [ν_h]
```

## Build

HYCOM 2.3 RELO MPI (`~/pyOCEAN/HYCOM-src`, sin modificar), copiado y compilado en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/hycom/build`.

Cambios hechos solo en la copia:
- **`config/generic-gnu-relo_mpi`:** se quitan `-mavx` y `-m64`, banderas de x86 que gfortran en arm64 no acepta.
- **EOS:** HYCOM la elige al compilar, y aquí es `CPP_EXTRAS="-DEOS_SIG0 -DEOS_7T"`. Sin estas banderas falla en `mod_incupd.F90` (`sig`, `sigver` sin definir).
- **Corrección de un error de HYCOM, clase (d)** (aprobada por Esteban el 2026-10-14):
  - `mod_hycom.F90:1656` y `:1674` ponen a cero `hnudge` y `salfac` cuando no hay marea (`tidnud = 0`, `tidflg ≤ 0`).
  - Pero `mod_tides.F90:217, 234-239` solo los reserva con la marea activa. Los dos arreglos se movieron a `mod_tides` en enero de 2025.
  - Resultado: cualquier corrida sin marea se cae con un segfault al terminar la inicialización.
  - `gate.sh` protege esas dos líneas con `if (allocated(...))`. Los demás usos ya están protegidos (`mod_barotp.F90:130-168`). No cambia ni la física ni el costo.
  - Se localizó con una copia compilada con `-O0 -g -fcheck=bounds` y `atos`.

## Caso

Lo escriben `inputs.py` y `blkdat.py` a partir de `cases.get("overflow")`:
- **Dominio:** 202 × 5 celdas con bordes de tierra, que dejan 200 × 3 celdas de océano. Δx = Δy = 1 km, f = 0, con el fondo de Petersen (2015) en cada fila.
- **Capas:** 100 capas híbridas (`nhybrd = 100`), con `dp0k` = 20 m. Como 100 × 20 m cubren los 2000 m de profundidad, las capas se quedan casi todas como z fijas: hybgen solo deja una interfaz isopicnal donde el agua la empuja hacia abajo (`hybgen.F90:446-460, 493-510`).
- **Densidades objetivo:** σ₀ de 7 términos, uniformes entre σ₀(20 °C) y σ₀(10 °C), con S = 35.
- **Estado inicial:** con `iniflg = 2`, desde 12 meses idénticos de archivos relax. Con `relax = 0`, nada se relaja.
- **Sin mezcla vertical ni viscosidad vertical** (`mlflag = 0`, `vertmx = 0`). Los demás modelos llevan ν_v = 1e-4 y pagan su solución implícita, así que esto favorece un poco el costo de HYCOM.
- **Sin difusión de interfaces ni fricción de fondo** (`thkdf2 = thkdf4 = 0`, `cb = 0`).
- **Viscosidad:** `veldf2 = ν_h/Δx`.
- **Orden de `blkdat.input`:** comprobado contra todas las lecturas `blkin*` de `blkdat.F90`. Con `nsigma = 0` no hay ninguna línea `ds0k`.

El tiempo es el del segundo bloque de `xctmrp`, que solo cubre el paso de tiempo (`timers.py`).

## Estabilidad: ν_h = 100 no termina

Con ν = 100, HYCOM se detiene en `cnuity` con grosores de capa negativos en el talud (`neg. dp`, x ≈ 43–55 km):
- a las 6.2 h con Δt 10 s;
- a las 6.8 h con Δt 5 s.

Así que no es un límite del paso de tiempo. **La causa no está diagnosticada.** No es el mismo mecanismo que en la híbrida de pyOCEAN (una capa se vacía en el paso lagrangiano) ni en MOM6 ρ (truncación de velocidad): aquí las capas son casi todas z, y faltan la difusión de interfaces y la fricción de fondo, en las que HYCOM suele apoyarse.

Con ν = 1000 termina las 12 h. Por eso la escalera de HYCOM usa ν = 1000, y no ν = 100 como los demás modelos. El costo por paso no depende de ν (30 ms por paso con Δt 10 y 20 s). El límite viscoso Δx²/8ν = 125 s no actúa entre 20 y 40 s. Una ν mayor solo puede ampliar el Δt máximo de HYCOM, así que la comparación le es favorable.

**Criterio de estabilidad.** Que termine con "normal stop" y que el rango de SSH no pase del doble del de Δt 10 s. Los demás modelos usan max\|u\| ≤ 2× la corrida de resultados, pero HYCOM no escribe velocidades sin archivos de salida.

| Δt baroclínico / barotrópico [s] | 1 proceso: s por hora simulada | Resultado |
|---|---|---|
| 10 / 1 | 10.9 | estable, sin `neg. dp` (SSH −0.89 a 0.18 m) |
| **20 / 2** | **5.41** | **estable, marginal:** un `neg. dp` de −8 cm a las 10.2 h, que HYCOM corrige (SSH −0.52 a 0.16 m) |
| 40 / 2 | — | 143 `neg. dp` desde las 10 h; se detiene (paso ~900) |
| 80 / 2, 160 / 2 | — | falla |

**Con Δt 20 s en 5 procesos:** 1.43 s/h, 3.8 veces más rápido que con 1 proceso, y la misma SSH.

**Reparto con Δt 20 s** [s de 12 h simuladas], 1 proceso / 5 procesos:

| Componente | 1 proceso | 5 procesos |
|---|---|---|
| momtum | 34.3 | 7.1 |
| tsadvc | 12.2 | 3.5 |
| hybgen | 8.6 | 2.2 |
| cnuity | 7.7 | 2.9 |
| barotp | 0.6 | 0.3 |
| total | 64.9 | 17.2 |

El momento se lleva más de la mitad del tiempo en 1 proceso. Con 5 procesos, `xctmrp` imprime los tiempos de un solo proceso, el último (mod_xc_mp.h:5326-5346), no los del más lento.
