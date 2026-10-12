# BayIW: sección 2-D del eje de la bahía, pyOCEAN contra MITgcm

**Fecha:** 2026-10-12. **Estado:** escrito por pedido de Esteban; falta su revisión.
**Rama:** `ale-plan-b`.
**Reemplaza el orden de** `2026-10-12-bayiw-baroclinic-benchmark-design.md` §5, que queda como referencia de configuración y métricas.

## 1. Por qué y nuevo orden

La BayIW completa (560 × 352 × 50) cuesta horas por corrida incluso después de la optimización del paso ALE (3.3 s/paso en z\*). Esteban pidió simulaciones más baratas que validen lo mismo. El nuevo orden:
1. **Sección 2-D a lo largo del eje de la bahía** (i = 280), pyOCEAN y MITgcm con la misma sección. Este spec.
2. **`internal_wave` de MITgcm** (marea interna sobre una pendiente, solución lineal). Antes necesita que pyOCEAN imponga una onda baroclínica que varía en el tiempo en la frontera abierta (una parte del alcance B del Plan C). Spec aparte.
3. **BayIW 3-D recortada** (la región de la bahía, unas 200 × 160 celdas, 25 niveles) contra MITgcm con el mismo recorte. Spec aparte.
4. **BayIW completa:** opcional; se decide al final.

**Duración de las corridas:** 60 h, la ventana forzada (decisión de Esteban).

## 2. La sección

- **Ubicación:** la columna i = 280 de la malla de MITgcm, de j = 0 (frontera sur, abierta) a j = 351 (costa norte, cerrada), con la bahía de j = 324 a 347. Pasa por las estaciones Gamma (215), C (265), Omega (315), Seiche2 (325) y Seiche (345).
- **Malla:** 1 × 352 × 50, con `dy` de MITgcm (5–13.3 km) y un `dx` igual al de i = 280 (5052.6 m). Sin frontera este–oeste: es un canal de una celda.
- **Profundidad:** la batimetría de la columna i = 280, recortada a 599.9 m, con bahía (`bahia_01_expand_bat`) y sin bahía (`nobahia_01_expand_bat`).
- **Estratificación:** dos capas (T_2layer) y lineal (linear_temp), con el perfil de la columna i = 280.
- **Viento:** el τ_y de `merid_local_bay_560x352.bin` en la columna i = 280, con la misma interpolación temporal (`wind_at`).
- **Rotación:** f₀ = 6.97e-5 s⁻¹. Con nx = 1 hay u y v sin gradientes en x, así que es una sección "2.5-D" con giro de Coriolis.
- **Frontera sur:** abierta, con Orlanski (CMAX 0.45) y esponja de 30 celdas solo en velocidad, como en 3-D.
- **Mezcla:** Ah = Kh = 100, ν = Kv = 1e-5, deslizamiento libre; EOS lineal de MITgcm (spec 3-D §3).

## 3. MITgcm de la sección

- **Experimento nuevo** en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mitgcm_bayiw2d/<case>_<bay>/`:
  - `code/` copiado de `BayIW_Rectan_2layer/code`, con `SIZE.h` sNx = 1, sNy = 352, nPx = nPy = 1, Nr = 50;
  - `packages.conf` y `OBCS_OPTIONS.h` iguales.
- **`data`:** el de la etapa 1 (startTime 0, viento periódico) con endTime = 216 000 s (60 h), `dumpFreq` = 900 s y `delXfile`/`delYfile`/`bathyFile`/`hydrogThetaFile`/`meridWindFile` de la sección. `data.obcs`: solo `OB_Jsouth`, con el mismo Orlanski y la misma esponja.
- **Archivos de entrada** escritos por un script (`validation/bayiw_2d/make_mitgcm_inputs.py`) a partir de los binarios originales (big-endian float64).
- **Compilación:** `genmake2 -mods ../code -optfile darwin_arm64_gfortran_esteban` sin MPI, en un solo proceso y un P-core. Cuesta minutos y las corridas, segundos.
- **Salidas:** las de MNC (η, T, U, V cada 900 s), leídas por `compare.py`.
- **Sin cambios al código de MITgcm.**

## 4. pyOCEAN de la sección

- **Constructor:** `validation/bayiw_2d/run_pyocean.py` reutiliza `validation/bayiw_baroclinic/bayiw.py` (entradas, parámetros, muestreo a profundidad fija, objetivos de la híbrida) para una sola columna i = 280.
- **Malla:** `create_cartesian_grid_stretched(dx=[5052.6], dy=dy_MIT)`.
- **Fronteras:** `open_boundaries=("south",)`.
- **Coordenadas:** z\* (50 `delR`) e híbrida (25 capas, 10 objetivos livianos).
- **Diagnóstico de mezcla:** apagado.
- **Paso de tiempo:** 30 s, igual que MITgcm; con 352 columnas el costo es despreciable. Así la comparación no mezcla diferencias de Δt.
- **Salidas cada 900 s:** η, T a 110/250 m y U/V a 110 m en las 5 estaciones de la sección, más la sección completa de T en los niveles RC de MITgcm.

## 5. Comparación y criterios

**Métricas:** las mismas del spec 3-D §6, adaptadas a la sección.
1. **Seiche:** Δη = η_bahía − η_sinbahía en Seiche y Seiche2 (0–27 h): período, rango, correlación.
2. **Ondas internas:** anomalía de T a 110 m y 250 m en las 5 estaciones, con y sin bahía, de 0 a 60 h: período dominante, amplitud rms, correlación y desfase.
3. **U/V a 110 m:** amplitud rms y correlación.
4. **Sección de T** a las 6, 24 y 60 h: error rms relativo de la anomalía; espesor de la interfaz en dos capas.
5. **Salud:** truncamientos, Σh = H + η, calor y RPE.

**Criterios:** más estrictos que en 3-D, porque ahora la configuración es idéntica (misma malla, mismo Δt, mismas entradas):
- seiche: período ±2 %, rango ±10 %, correlación ≥ 0.95;
- ondas internas: período ±5 %, amplitud ±20 %, correlación ≥ 0.8 en las estaciones de la bahía y la boca (Seiche, Seiche2, Omega, C);
- efecto de la bahía: mismo signo y orden de magnitud;
- z\* frente a híbrida: se reportan sin umbral.

Toda diferencia se clasifica (a)–(d); si es (a), me detengo y se lo reporto a Esteban antes de seguir.

## 6. Entregables

- `validation/bayiw_2d/`: `make_mitgcm_inputs.py`, `run_mitgcm.sh`, `run_pyocean.py`, `compare.py` (reutiliza las métricas de `validation/bayiw_baroclinic/compare.py`) y `README.md` con las tablas, la clasificación y "qué debe mejorar pyOCEAN".
- Tests baratos en `tests/test_bayiw_2d.py`:
  - la sección leída coincide con la columna i = 280 de los binarios;
  - los archivos de MITgcm tienen las formas y el orden de bytes correctos;
  - el modelo de la sección se construye y da 1 paso.
- Una fila en el marcador del roadmap; actualización de `CLAUDE.md` y de la memoria.

## 7. Fuera de alcance

- los pasos 2–4 del §1 (specs aparte);
- el caso realista (necesita salinidad);
- cambios al código de MITgcm.
