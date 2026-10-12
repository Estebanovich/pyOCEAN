# Benchmark baroclínico de BayIW contra MITgcm

**Fecha:** 2026-10-12. **Estado:** diseño aprobado por Esteban en la conversación; falta revisar el texto escrito.
**Rama:** `ale-plan-b`.
**Depende de:** el modo ALE con fronteras abiertas (Plan C, commit 8a8dba1; spec `2026-10-12-ale-open-boundaries-design.md`).

## 1. Objetivo

Primer caso realista de pyOCEAN baroclínico: reproducir las corridas de MITgcm de la tesis de Esteban (bahía rectangular con estratificación de dos capas y lineal, con y sin bahía) con el modo ALE, primero en z\* y luego en la coordenada híbrida. El benchmark mide si la ventaja de ALE sobre topografía abrupta (escalón de la boca 600 → 164 m: 5e-12 m/s en z\* contra 7.3 cm/s en σ) se sostiene en un caso forzado de ondas internas. Sigue las reglas de benchmark de `CLAUDE.md`: clasificar cada diferencia como (a) error nuestro, (b) física faltante, (c) elección numérica o (d) error del modelo de referencia; no calibrar a un solo caso; terminar con la lista "qué debe mejorar pyOCEAN" y actualizar el marcador (§7 de `docs/design/vision_roadmap.md`).

**Casos (elegidos por Esteban):** dos capas y lineal, cada uno con bahía y sin bahía.

## 2. Referencias de MITgcm

| Caso | Directorio | Estratificación | t_ref |
|---|---|---|---|
| Dos capas | `/Volumes/ESTEBAN_MIT/MITgcm/verification/BayIW_Rectan_2layer/` | 22.77 °C sobre 18.52 °C, interfaz a ≈116 m (Δρ ≈ 0.85 kg m⁻³, c₁ ≈ 0.9 m s⁻¹) | 11.4 °C |
| Lineal | `/Volumes/Esteban_SSD/Maestría/MITgcm/verification/BayIW_Rectan_linear/` | 33.6 °C en superficie a 3.5 °C a 577 m (N ≈ 0.010 s⁻¹, c₁ ≈ 1.9 m s⁻¹) | 33.6 °C |

- **Archivos de entrada:**
  - T inicial: `input/T_2layer_50z_560x352.bin` y `input/linear_temp_50zlev_560x352.bin`;
  - batimetría: `bahia_01_expand_bat.bin` / `nobahia_01_expand_bat.bin`;
  - malla: `*_dx.bin` / `*_dy.bin`;
  - viento: `merid_local_bay_560x352.bin`.
- **Series de comparación** en `/Volumes/ESTEBAN_MIT/MITgcm/verification/Output_IWBay/npz_out/`, cada 900 s de 0 a 120 h:
  - `<case>_series.npz` (η con bahía, sin bahía y su diferencia en las estaciones Seiche, Seiche2, A, B, C, Gamma, Omega);
  - `<case>_Temp_T110m_series.npz`, `_T250m_series.npz` y `<case>_UV_UV110m_series.npz`;
  - `<case>_Temp_zoom_z110m_{bay,nobay}.npz` (campo a 110 m, 352 × 300);
  - `<case>_secThetaAnom_yz_Gamma.npz` (sección YZ);
  - `<case>_*_espectros.npz`.
- En dos capas también están los campos completos: `diag_rho_{bay,nobay}.glob.nc` (RHOAnoma, 480 × 50 × 352 × 560).

## 3. Configuración de pyOCEAN

**Correspondencia con MITgcm** (`run_expand/data`, `data.obcs`):

| Elemento | MITgcm | pyOCEAN |
|---|---|---|
| Malla | 560 × 352 estirada (`delX/delYFile`), 5 km cerca de la bahía y hasta 13.3 km | `create_cartesian_grid_stretched`, f₀ = 6.97e-5 s⁻¹ |
| Batimetría | recortada a Σ delR = 599.9 m (intencional, confirmado por Esteban) | igual que `validation/bayiw_seiche/` |
| Vertical (z\*) | 50 niveles `delR` (1 m arriba, 45.8 m abajo) | `regrid="zstar"`, `dz_top` = `delR` desde la superficie |
| EOS | lineal, `tAlpha` = 2e-4, `rhoNil` = 999.8, `tRef` uniforme | `rho0=999.8`, `t_coef` = αρ₀ = 0.19996, `t_ref` = tRef |
| Viscosidad y difusión | `viscAh` = `diffKhT` = 100, `viscAr` = `diffKrT` = 1e-5, deslizamiento libre, sin arrastre de fondo | `ah=100`, `kh=100`, `nu_const=1e-5`, `kv=1e-5`, `z0=None`, `closure="constant"` |
| Gradiente de presión | hidrostático en z | `pgf="fv"`, `rho_ref="auto"` |
| Fronteras | S, W, E abiertas, N cerrada; Orlanski con `CMAX` = 0.45; esponja de 30 celdas solo en velocidad (τ 60 s en el borde → 1200 s adentro); `useOBCSbalance` | `open_boundaries=("south","west","east")`, `orlanski_cmax=0.45`, `sponge_cells=30`, `sponge_tau=(1200.0, 60.0)`, `sponge_tracer=False`; Flather |
| Viento | `meridWindFile` en la etapa 1 (0–60 h), sin viento en la etapa 2 | `wind_at` de `validation/bayiw_seiche/run_pyocean.py` |
| Paso de tiempo | 30 s | 120 s o 60 s (§5, fase 0) |

**Diferencias conocidas, clasificadas desde antes:**
- **(b) Ajuste convectivo** (`ivdc_kappa` = 1) que pyOCEAN no tiene. Con una estratificación estable y viento moderado casi no debería activarse; si aparece inestabilidad estática, se reporta.
- **(c) Balance de masa** en la frontera (`useOBCSbalance`) contra Flather.
- **(c) Momento:** vectorial invariante con KE Godunov y transporte de grosor PPM, contra la forma advectiva de MITgcm.

**Coordenada híbrida (fase 3):**
- 25 capas;
- `dz_top` = los primeros 25 `delR` de MITgcm (1.0 a 6.5 m, Σ = 74.8 m): superficie en z\*, y todo lo que está debajo de ≈75 m es isopicno, incluida la interfaz de dos capas a 116 m;
- objetivos de densidad:
  - **dos capas:** cada masa de agua en el centro de su intervalo de densidad, para que ninguna densidad caiga exactamente sobre un objetivo (el empate del lock exchange de 21 capas);
  - **lineal:** `target_densities` del perfil, con un margen de medio intervalo en los extremos.

## 4. Corredor y salidas

**`validation/bayiw_baroclinic/run_pyocean.py`:**
- **Argumentos:** `case 2layer|linear`, `bay|nobay`, `--coord zstar|hybrid`, `--dt`, `--hours`.
- **Reutiliza** la lectura de MITgcm, la malla, las estaciones y `wind_at` del corredor barotrópico (importados, no copiados).
- **Muestreo a profundidad fija:** T, U y V a 110 m y 250 m se interpolan en z entre los centros de capa, porque las capas se mueven. η se toma directo.
- **Salida cada 900 s** en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_<case>/<coord>_<bay>/`:
  - series en las 7 estaciones;
  - campo de T a 110 m en la ventana zoom de MITgcm (`X_zoom`, `Y_zoom` de su npz);
  - sección YZ de Gamma;
  - diagnósticos de salud: truncamientos, max|u|, contenido de calor y RPE.
- **Estado simple cada 6 h** (`np.savez` de η, h, T, u, v y el tiempo) en el mismo directorio, para retomar a mano si una corrida se cae. El restart ALE bit a bit es del Plan D. Los `.npz` no se versionan.
- **Hilos:** 5 P-cores (`NUMBA_NUM_THREADS=5`, `set_threads(5)`), una corrida a la vez.

**`validation/bayiw_baroclinic/compare.py`:** lee las salidas de pyOCEAN y los npz de MITgcm y escribe tablas y figuras (§6).

## 5. Fases y reglas de avance

**Fase 0, preparación:**
1. **Fronteras abiertas con rotación** (pendiente de la revisión del Plan C). Tests con f = 1e-4, en z\* y en la híbrida:
   - un trazador uniforme se mantiene y el volumen sale;
   - la estratificación en reposo se queda en reposo.
2. **Tests baratos del corredor:**
   - el muestreo a una profundidad fija es exacto para un perfil lineal en capas arbitrarias;
   - la lectura de las referencias da las formas esperadas;
   - el viento coincide con el barotrópico.
3. **Prueba de Δt:** las primeras 12 h de dos capas con bahía en z\*, con Δt de 60 y 120 s. Si la diferencia rms de T a 110 m y de η en las 7 estaciones es menor que el 5 % de la amplitud rms de la señal, se usa 120 s (≈8.4 h por corrida); si no, 60 s (≈17 h).

**Fase 1:** dos capas en z\*, con y sin bahía, y comparación.
**Fase 2:** lineal en z\*, con y sin bahía, y comparación.
**Fase 3:** la híbrida en los dos casos, con y sin bahía (≈4.7 h por corrida), y la tabla z\* contra híbrida contra MITgcm.

**Regla de avance:**
- Después de cada fase se comparan los resultados.
- Si un criterio de §6 falla por (a) un error nuestro, me detengo, lo diagnostico y se lo reporto a Esteban antes de gastar más corridas.
- Si falla por (b) o (c), se documenta y se sigue.

**Costo:** ≈55 h de cómputo con Δt = 120 s.

## 6. Comparación y criterios de éxito

**Ventanas:** forzada (0–60 h) y libre (60–120 h).

**Métricas:**
1. **Seiche barotrópico:** Δη = η_bahía − η_sinbahía en Seiche y Seiche2: período, rango, correlación y e-folding, como en `validation/bayiw_seiche/compare.py`.
2. **Ondas internas:** anomalía de T a 110 y 250 m en las 7 estaciones, con y sin bahía:
   - período dominante (espectro);
   - amplitud rms;
   - correlación y desfase.
3. **Corrientes:** U y V a 110 m en las estaciones: amplitud rms y correlación.
4. **Estructura espacial:** campo de T a 110 m (zoom) y sección YZ de Gamma a las 6, 24, 60 y 90 h: error rms relativo y figuras lado a lado.
5. **Salud:** cero truncamientos CFL, Σh = H + η, contenido de calor (cambia solo por las fronteras abiertas) y ganancia de RPE (la mezcla espuria de la interfaz).

**Criterios.** Son metas, no ajustes; si uno falla, se clasifica (a)–(d):
- **Seiche:** período ±2 %, rango ±15 %, correlación ≥ 0.9 entre 0 y 27 h.
- **Ondas internas** (T a 110 m, ventana forzada):
  - período dominante ±5 %;
  - amplitud rms ±25 %;
  - correlación ≥ 0.7 en Seiche, Seiche2, Omega y C.

  En la ventana libre solo cuentan el período y la amplitud: la fase de las ondas internas diverge con el tiempo por diferencias numéricas pequeñas y eso no es un error.
- **Efecto de la bahía:** ΔT a 110 m (con bahía menos sin bahía) con el mismo signo y orden de magnitud que MITgcm en las estaciones de la bahía y de la boca.
- **Híbrida frente a z\*:** se reportan la nitidez de la interfaz (espesor entre 10 % y 90 % de ΔT en la sección de Gamma), la RPE y las métricas 1–4. No hay umbral: es un resultado.

## 7. Entregables

- `validation/bayiw_baroclinic/`: `run_pyocean.py`, `compare.py`, `README.md` con las tablas, la clasificación de las diferencias y la lista "qué debe mejorar pyOCEAN".
- Tests de la fase 0 en `tests/`.
- Fila "BayIW baroclínico" en §7 de `docs/design/vision_roadmap.md`, `CLAUDE.md` y la memoria del proyecto.

## 8. Fuera de alcance

- el caso realista (climatología de febrero): necesita salinidad en el modo ALE;
- σ euleriano: ya se sabe que falla en la boca (7.3 cm s⁻¹ de flujo espurio en el escalón);
- restart ALE bit a bit (Plan D);
- las variantes `linear_2xwind`, `uniform` y `pile`;
- remapeo de la salida a niveles z completos (solo 110 m, 250 m y la sección).
