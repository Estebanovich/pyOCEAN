# Benchmark de cuatro modelos, parte A: casos de Ilicak (2012) y seamount

- **Spec:** `docs/superpowers/specs/2026-10-13-ilicak-four-model-benchmark-design.md`
- **Plan:** `docs/superpowers/plans/2026-10-13-ilicak-benchmark-part-a.md`

## Qué es

Cuatro casos baratos, definidos una sola vez en `cases.py`. Cada modelo los corre desde esa definición y todos se miden con las mismas métricas (`metrics.py`).

| Caso | Definición | Malla | Δt de resultados | Duración | ν_h [m² s⁻¹] |
|---|---|---|---|---|---|
| Lock exchange | Ilicak §3.1; T 5/30 °C como Petersen (2015) y MOM6 | 128 × 20 | 10 s | 17 h | 0.1, 10, 100 |
| Overflow | Ilicak §4 con la plataforma de Petersen (ec. 24–25) | 200 × 100 | 10 s | 40 h | 1, 100, 1000 |
| Onda interna | Petersen ec. 26–27, alta amplitud (A = 2 °C) | 50 × 20 | 300 s | 200 días | 0.01, 15, 150 |
| Seamount | Beckmann & Haidvogel (1993), el de `validation/seamount` | 64 × 48 × 20 | 600 s | 10 días | 0 |

**Física común:**
- EOS lineal ρ = 1000 − 0.2 (T − 0) y g = 9.81;
- sin difusión explícita de trazadores;
- ν_v = 1e-4, sin fricción de fondo, deslizamiento libre y paredes cerradas.

**Configuraciones.** Cada modelo usa sus propios esquemas:

| Modelo | Coordenadas | Esquemas |
|---|---|---|
| pyOCEAN | z\*, σ, híbrida | vector-invariante con KE Arakawa, WENO5-Z, PPM_H4, continuidad PPM, PGF de volumen finito; transporte baroclínico de capa centrado (θ_bc = ½; la híbrida θ = 0.6, punto 9) |
| MOM6 (`repro -O2`, 1 proceso) | z\*, ρ (`layer_ref` con los mismos objetivos que la híbrida); σ en el seamount | PPM:H3, KE_ARAKAWA, PPM_H4 |
| CROCO 2.1.3 (serial) | σ uniforme (`NEW_S_COORD`, hc = 1e16) | los de REGIONAL: UV_HADV_UP3, UV_VADV_SPLINES, TS_HADV_RSUP3, TS_VADV_SPLINES |
| MITgcm (serial) | z\* con celdas parciales (hFacMin 0.1) | OS7MP, momento en forma de flujo |

En el seamount entran pyOCEAN σ y z\*, MOM6 σ y z\*, CROCO σ y MITgcm z\*.

**Diferencias de configuración, todas de clase (c):**
- **CROCO:**
  - RSUP3 lleva una difusión biarmónica explícita |u|Δx³/12 con mezcla rotada (`cppdefs_dev.h:423-436`);
  - `RESET_RHO0` es forzado: ρ₀ es la media del volumen, lo que cambia g/ρ₀ en 3.5e-3–4e-3.
- **MITgcm:** momento en forma de flujo y celdas parciales.
- **pyOCEAN, seamount:** usa el perfil de referencia `rho_ref = "auto"`. Sin él, el Jacobiano en σ daba 42 cm/s (`validation/seamount/README.md`).
- **Procesos:** MOM6, CROCO y MITgcm corren en 1 proceso (con NJ = 1, MOM6 solo admite 1) y pyOCEAN en 5 hilos. También se midió pyOCEAN con 1 hilo.

## Cómo reproducirlo

```bash
for c in lock overflow iw seamount; do validation/ilicak_benchmark/run_croco.sh build $c; validation/ilicak_benchmark/run_mitgcm.sh build $c; done
python validation/ilicak_benchmark/campaign.py results [caso]     # 69 corridas, una a la vez
python validation/ilicak_benchmark/campaign.py cost [caso]        # escalera de Δt ×1, 2, 4, 8, 16 (1 h; 1 día en IW y seamount)
python validation/ilicak_benchmark/compare.py [caso]              # tablas
```

- Las salidas van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/<caso>/<modelo>_<coord>_nu<ν>/series.npz`, en un formato común con las capas desde la superficie.
- Una corrida completa tarda de 2 s (CROCO, lock exchange) a 7 min (MOM6, overflow).

**Métricas:**
- **RPE** (Winters et al. 1995; Ilicak ec. 1–2): con `diagnostics.mixing.reference_potential_energy`, sobre el volumen real de cada celda y la hipsometría exacta del fondo de cada modelo.
- **dRPE/dt** por unidad de área, en tres ventanas: la media de 15–17 h en el lock exchange (el spec dice "a las 17 h"), 13–37 h en el overflow y días 10–100 en la onda interna.
- **Rango de T** en celdas de más de 1 mm y **extremos nuevos** (`t_excess`): la mayor excursión fuera del rango inicial en celdas de al menos 1 m y la fracción del volumen que sale del rango por más de 0.01 °C.
- **Lock exchange:** frentes con umbrales de 1 m (punta) y 5 m (cuerpo), ajustados entre 2 y 8 h, divididos entre la velocidad de Benjamin U = 0.496 m/s.
- **Overflow:** centroide del agua con T < 15 °C, la métrica del spec, y centroide del déficit de calor (20 − T)⁺. Esta segunda se añadió porque la primera queda indefinida cuando el agua densa se mezcla por encima de 15 °C.
- **Onda interna:** amortiguamiento de la oscilación (`oscillation_ratio`): desviación estándar del desplazamiento de la isoterma de 15.1 °C en x₀ en los últimos 20 días, dividida entre la de los primeros 20. La desviación estándar quita el desplazamiento medio, de ~60 m, que deja el abultamiento al extenderse; la primera versión usaba el rms y medía sobre todo ese desplazamiento.
- **max|u|** en caras de al menos 1 cm: las velocidades en capas vacías no llevan masa.

**Re_Δ = UΔx/ν_h:**
- lock exchange (U = 0.496 m/s, Δx 500 m): 2480, 25 y 2.5;
- overflow (U = 1.57 m/s, Δx 1 km): 1570, 16 y 1.6;
- onda interna (U ≈ 0.37 m/s, Δx 5 km): 1.9e5, 120 y 12.

## Resultados

### Lock exchange (17 h)

| ν_h | Modelo | ΔRPE/RPE₀ | dRPE/dt [W m⁻²] | Rango de T | Frentes / U (punta densa, punta ligera, cuerpo denso, cuerpo ligero) |
|---|---|---|---|---|---|
| 0.1 | pyOCEAN z\* (= σ) | 6.67e-5 | 3.86e-3 | 5–30 | 0.88, 0.87, 0.88, 0.86 |
| 0.1 | pyOCEAN híbrida | 6.2e-8 | 4.2e-6 | **3.24–30.1** | 0.97, 0.96, 0.99, 0.97 |
| 0.1 | MOM6 z\* | 6.75e-5 | 3.91e-3 | 5–30 | 0.87, 0.87, 0.88, 0.86 |
| 0.1 | MOM6 ρ | 1.46e-5 | 9.4e-4 | 5–30 | 0.85, 0.96, 0.68, 0.94 |
| 0.1 | CROCO σ | 1.91e-5 | 2.13e-3 | **−9.3–39.1** | 0.95, 0.96, 0.96, 0.95 |
| 0.1 | MITgcm z\* | 6.89e-5 | 4.28e-3 | 4.93–30.1 | 0.92, 0.91, 0.91, 0.90 |
| 10 | pyOCEAN z\* (= σ) | 5.73e-5 | 3.29e-3 | 5–30 | 0.89, 0.89, 0.89, 0.89 |
| 10 | pyOCEAN híbrida | 6.7e-8 | 4.3e-6 | **4.02–30.6** | 1.01, 0.98, 1.00, 1.00 |
| 10 | MOM6 z\* | 5.81e-5 | 3.36e-3 | 5–30 | 0.88, 0.89, 0.89, 0.89 |
| 10 | MOM6 ρ | 1.57e-5 | 8.9e-4 | 5–30 | 0.96, 0.88, 0.90, 0.71 |
| 10 | CROCO σ | 1.71e-5 | 1.71e-3 | **−8.8–38.9** | 0.96, 0.96, 0.96, 0.96 |
| 10 | MITgcm z\* | 5.14e-5 | 2.88e-3 | 4.99–30.0 | 0.93, 0.93, 0.93, 0.93 |
| 100 | pyOCEAN z\* (= σ) | 1.90e-5 | 1.08e-3 | 5–30 | 0.95, 0.95, 0.95, 0.96 |
| 100 | pyOCEAN híbrida | 7.5e-8 | 4.5e-6 | **4.14–31.0** | 1.04, 1.04, 1.03, 1.03 |
| 100 | MOM6 z\* | 1.87e-5 | 1.07e-3 | 5–30 | 0.95, 0.95, 0.95, 0.96 |
| 100 | MOM6 ρ | 1.41e-5 | 9.2e-4 | 5–30 | 0.96, 0.84, 0.44, 0.53 |
| 100 | CROCO σ | 4.0e-6 | 4.8e-4 | **−7.3–37.0** | 0.99, 0.99, 0.98, 0.98 |
| 100 | MITgcm z\* | 1.37e-5 | 7.7e-4 | 4.99–30.0 | 0.98, 0.99, 0.99, 0.99 |

### Overflow (40 h)

| ν_h | Modelo | ΔRPE/RPE₀ | dRPE/dt [W m⁻²] | Rango de T | Centroide frío: x [km], profundidad [m] |
|---|---|---|---|---|---|
| 1 | pyOCEAN z\* | 3.55e-5 | 5.26 | 7.02–20.1 | 117, 1280 |
| 1 | pyOCEAN σ | 3.19e-5 | 4.17 | 9.75–21.6 | 141, 1160 |
| 1 | pyOCEAN híbrida | — se detiene: "layer emptied in the Lagrangian step (reduce dt)" | | | |
| 1 | MOM6 z\* | 3.14e-5 | 2.12 | 10–20 | 83, 980 |
| 1 | MOM6 ρ | — se detiene: velocidad truncada demasiadas veces (~16 h) | | | |
| 1 | CROCO σ | 3.0e-6 | 0.94 | **3.2–25.8** | 146, 1600 |
| 1 | MITgcm z\* | — se detiene: `CALC_R_STAR`, rStarFac demasiado pequeño (~11 h, u hasta 13 m/s) | | | |
| 100 | pyOCEAN z\* | 1.41e-5 | 1.86 | 6.35–20.0 | 154, 1460 |
| 100 | pyOCEAN σ | 1.49e-5 | 1.69 | 9.76–20.2 | 154, 1410 |
| 100 | pyOCEAN híbrida | — se detiene: "layer emptied in the Lagrangian step" | | | |
| 100 | MOM6 z\* | 3.24e-5 | 4.36 | 10–20 | 137, 944 |
| 100 | MOM6 ρ | — se detiene: velocidad truncada demasiadas veces | | | |
| 100 | CROCO σ | 8.1e-7 | 0.59 | **1.3–25.4** | 148, 1670 |
| 100 | MITgcm z\* | 1.34e-5 | 1.44 | 9.59–20.1 | 124, 1370 |
| 1000 | pyOCEAN z\* | 4.9e-6 | 0.58 | 5.26–20.0 | 150, 1770 |
| 1000 | pyOCEAN σ | 5.8e-6 | 0.68 | 9.89–20.0 | 153, 1740 |
| 1000 | pyOCEAN híbrida | 4.7e-11 | 7.2e-6 | **8.72–33.1** (max\|u\| 47 m/s) | 142, 1960 |
| 1000 | MOM6 z\* | 1.34e-5 | 1.66 | 10–20 | 154, 1330 |
| 1000 | MOM6 ρ | 1.2e-6 | 0.19 | 10–20 | 101, 1770 |
| 1000 | CROCO σ | −5.3e-6 | −0.25 | **2.9–25.7** | 150, 1800 |
| 1000 | MITgcm z\* | 4.8e-6 | 0.52 | 9.63–20.1 | 148, 1720 |

En las primeras 3 h los tres modelos z\* coinciden casi exactamente: grosor del agua densa, η y max\|u\| de ~2.3 m/s. En la caída por el talud todos llegan a 3–10 m/s; la caída libre √(2 g′ 1500 m) da ~8 m/s.

### Onda interna (200 días)

| ν_h | Modelo | ΔRPE/RPE₀ | dRPE/dt [W m⁻²] (días 10–100) | Rango de T | Oscilación final / inicial |
|---|---|---|---|---|---|
| 0.01 | pyOCEAN z\* (= σ) | 3.25e-6 | 3.14e-4 | 10.10–19.84 | 0.74 |
| 0.01 | pyOCEAN híbrida | 9.1e-8 | 2.5e-6 | 10.10–19.84 | **0.45** |
| 0.01 | MOM6 z\* | 3.70e-6 | 3.81e-4 | 10.10–19.84 | 0.67 |
| 0.01 | MOM6 ρ | 6.1e-7 | 4.0e-5 | 10.10–19.84 | 0.65 |
| 0.01 | CROCO σ | −8.5e-8 | 7.2e-6 | **10.01–19.96** | 0.68 |
| 0.01 | MITgcm z\* | 2.95e-6 | 3.41e-4 | 10.10–19.84 | 0.72 |
| 15 | pyOCEAN z\* (= σ) | 1.71e-6 | 1.85e-4 | 10.10–19.84 | 0.58 |
| 15 | pyOCEAN híbrida | 8.8e-8 | 2.2e-6 | 10.10–19.84 | **0.43** |
| 15 | MOM6 z\* | 2.21e-6 | 2.57e-4 | 10.10–19.84 | 0.55 |
| 15 | MOM6 ρ | 1.4e-7 | 1.1e-5 | 10.10–19.84 | 0.65 |
| 15 | CROCO σ | −7.4e-8 | 6.3e-6 | **10.01–19.96** | 0.60 |
| 15 | MITgcm z\* | 1.42e-6 | 1.76e-4 | 10.10–19.84 | 0.56 |
| 150 | pyOCEAN z\* (= σ) | 6.5e-7 | 6.2e-5 | 10.10–19.84 | 0.26 |
| 150 | pyOCEAN híbrida | 8.5e-8 | 1.9e-6 | 10.10–19.84 | 0.25 |
| 150 | MOM6 z\* | 7.0e-7 | 7.0e-5 | 10.10–19.84 | 0.25 |
| 150 | MOM6 ρ | 3.2e-8 | 2.8e-6 | 10.10–19.84 | 0.26 |
| 150 | CROCO σ | −5.6e-8 | 3.4e-5 | **10.01–19.96** | 0.26 |
| 150 | MITgcm z\* | 8.6e-7 | 8.4e-5 | 10.10–19.84 | 0.24 |

### Seamount (10 días, sin viscosidad)

| Modelo | max\|u\| a 1 / 2 / 10 días [cm/s] |
|---|---|
| pyOCEAN σ (PGF de volumen finito + ρ_ref) | 0.17 / 0.20 / 0.22 |
| pyOCEAN z\* | 0.02 / 0.02 / 0.08 |
| MOM6 σ | 79 / 83 / 154, creciendo |
| MOM6 z\* | se detiene: velocidad truncada desde el día 1, también con Δt 300 s y con `BOUND_CORIOLIS` + `CORIOLIS_EN_DIS` + `MIN_THICKNESS` 0.1 |
| CROCO σ | 3.6 / 6.5 / 36, creciendo |
| MITgcm z\* | 0.00 (exacto: isopicnas planas en niveles z) |

Beckmann & Haidvogel (1993) obtuvieron ~1 cm/s a 10 días. El ejemplo `seamount` de MOM6 sí corre, pero añade KH = 1000, KHTH = 500 y fricción, que este caso excluye.

### Costo

**Cómo se mide.** Segundos de reloj por hora simulada, en el bucle principal sin la inicialización:
- **pyOCEAN:** después de los 5 primeros pasos, que incluyen la compilación de Numba;
- **MOM6:** el reloj "Main loop";
- **MITgcm:** `MAIN_DO_LOOP`;
- **CROCO:** `/usr/bin/time` de toda la corrida. Su resolución es de 0.01 s, así que los valores de CROCO por debajo de ~0.05 s/h tienen ±50 %.

**Comparación principal: al Δt máximo estable y con 1 proceso en todos los modelos** (spec §A2). Se usa la escalera Δt × 1, 2, 4, 8, 16 sobre 2 h (lock exchange), 12 h (overflow, más allá de la caída por el talud) y 48 h (onda interna y seamount), con la ν_h intermedia.

Un escalón cuenta como estable si T y u son finitas y max\|u\| no pasa del doble de la corrida de resultados en el mismo tramo. "≥" indica que se llegó al tope de la escalera.

| Caso | pyOCEAN z\* | MOM6 z\* | CROCO σ | MITgcm z\* | HYCOM (solo costo) |
|---|---|---|---|---|---|
| Lock exchange: Δt máx [s] / s por hora | ≥ 160 / 0.045 | ≥ 160 / 0.076 | ≥ 160 / 0.015 | ≥ 160 / 0.106 | — |
| Overflow | **80 / 0.40** | 20 / 4.28 | 10 / 0.94 | 10 / 4.69 | 20 / 5.75 (ν = 1000) |
| Onda interna | 2400 / 0.0017 | 2400 / 0.0025 | 2400 / 0.0008 | 2400 / 0.0038 | — |
| Seamount (MOM6: σ) | 600 / 0.20 | 1200 / 0.19 | 600 / 0.12 | 1200 / 0.027 | — |

**Al Δt de resultados** (10 s, 10 s, 300 s y 600 s). pyOCEAN con 5 hilos y con 1 hilo; los demás con 1 proceso:

| Caso | pyOCEAN z\* (5 hilos / 1 hilo) | MOM6 z\* | CROCO σ | MITgcm z\* | HYCOM |
|---|---|---|---|---|---|
| Lock exchange | 0.93 / 0.71 | 1.16 | 0.13 | 0.75 | — |
| Overflow | 3.02 / 3.13 | 10.3 | 1.05 | 5.06 | 11.2 (ν = 1000) |
| Onda interna | 0.022 / 0.013 | 0.017 | 0.0024 | 0.013 | — |
| Seamount | 0.15 / 0.20 | 0.39 (σ) | 0.13 | 0.051 | — |

Notas:
- Las otras coordenadas cuestan casi lo mismo que z\* dentro de cada modelo (`compare.py`).
- La híbrida de pyOCEAN y MOM6 ρ no tienen costo en el overflow, porque sus corridas de resultados con la ν_h intermedia no terminan.
- **HYCOM** (`hycom/README.md`) es solo costo: 100 capas híbridas con `hybgen`, EOS σ₀ de 7 términos y sin mezcla explícita. Con 100 × 20 m de `dp0k`, sus capas son casi todas z. Con ν = 100 tampoco termina (`neg. dp` en el talud a las 6–7 h, también con Δt 5 s; causa sin diagnosticar), así que se mide con ν = 1000. El costo por paso no depende de ν. Con Δt 20 s tiene un evento de grosor negativo, así que es un aprobado marginal; sin ninguno, el Δt máximo sería 10 s (11.2 s/h). Con 5 procesos cuesta 2.84 s/h.
- No se midió el reparto por componente que pedía el spec: los casos 2-D son tan pequeños que domina el trabajo fijo por paso.

## Lectura y clasificación

1. **pyOCEAN z\* = MOM6 z\* en el lock exchange** en las tres viscosidades: RPE dentro del 1.6 % y frentes iguales. MITgcm queda a ±30 % en RPE y es un poco más rápido. Ninguno de los tres crea extremos apreciables: ≤ 2e-4 °C en pyOCEAN y ≤ 0.09 °C en MITgcm. **Coinciden.**
2. **Velocidad del frente contra Benjamin.** pyOCEAN z\* queda dentro de ±10 % con ν = 10 y 100 (0.89–0.96 U). Con ν = 0.1 baja a 0.86–0.88 U, igual que MOM6 z\*. Ilicak (Fig. 3) también ve frentes lentos en modelos hidrostáticos con esta resolución. **(c)**
3. **Re_Δ:** los modelos z\* reducen dRPE/dt entre 3.6 y 5.6 veces al pasar de Re_Δ 2480 a 2.5: pyOCEAN 3.6, MOM6 3.7 y MITgcm 5.6. Ilicak describe un orden de magnitud entre Re_Δ > 100 y < 5. pyOCEAN se comporta como los demás. **(c)**
4. **Hipótesis previa 1, que pyOCEAN mezclaría más que CROCO con Re_Δ alto: no se puede contestar con este CROCO.**
   - Sus esquemas regionales crean extremos nuevos de T: hasta 14 °C fuera de rango en el 11–18 % del volumen en el lock exchange, y 6.8–8.7 °C en el 25–40 % en el overflow.
   - Su RPE baja (es negativa en la onda interna y en el overflow con ν = 1000) y no es comparable (Ilicak, Fig. 7).
   - Para responderla hace falta CROCO con WENO5. **(c)**, pendiente.
5. **Hipótesis previa 2, que la híbrida e isopicnal mezclarían mucho menos que z\*: se confirma.**
   - **Lock exchange:** la híbrida de pyOCEAN mezcla ~1000 veces menos que z\* y ~200 veces menos que MOM6 ρ. Tiene los mejores frentes (0.96–1.04 U); MOM6 ρ tiene el cuerpo ligero lento (0.44–0.94 U).
   - **Onda interna:** mezcla 7.7–36 veces menos que z\*, y entre 0.15 y 2.7 veces lo de MOM6 ρ. La comparación con z\* favorece algo a la híbrida, cuya onda todavía se amortigua más (punto 9).
   - **Su RPE sí es comparable**, a diferencia de la de CROCO: la T fuera de rango (punto 6) está solo en capas de menos de 1 m. En celdas de al menos 1 m la excursión es ≤ 1.8e-5 °C en el lock exchange y ≤ 3e-4 °C en el overflow.
6. **La híbrida de pyOCEAN crea T fuera de rango en capas delgadas: (a).**
   - Lock exchange: T de 3.2 a 31 °C, en capas de ~1 mm.
   - Overflow con ν = 1000: hasta 33.1 °C, y max\|u\| de 47 m/s en caras de al menos 1 cm.

   MOM6 ρ mantiene el rango exacto. Es el problema conocido de T fuera de rango en capas de 1 mm del plan de la híbrida, ahora con un reproductor de 1 h.
7. **La híbrida de pyOCEAN no completa el overflow con ν ≤ 100: (a) de robustez.** Se detiene porque una capa se vacía en el paso lagrangiano. MOM6 ρ también falla en los mismos dos casos, por truncación de velocidad, así que la isopicnal con Re_Δ alto en esta pendiente es difícil para los dos. Pero pyOCEAN debería limitar el flujo saliente de la capa en lugar de detenerse.
8. **Overflow en z\*:**
   - **Mezcla y descenso:** pyOCEAN z\* ≈ MITgcm z\*. Con ν = 100, dRPE/dt es 1.85 contra 1.44 W m⁻²; con ν = 1000, 0.58 contra 0.52, y el centroide frío 150 km / 1770 m contra 148 km / 1720 m.
   - **MOM6 z\*** mezcla 2.4–2.9 veces más con ν = 100 y 1000, y no le queda agua bajo 15 °C. Con ν = 1 mezcla menos de la mitad que pyOCEAN (2.12 contra 5.26).
   - **Con ν = 1** terminan pyOCEAN z\*, MOM6 z\* y CROCO; MITgcm se rompe a las ~11 h.
   - **pyOCEAN z\* crea extremos nuevos de T: (a)/(c).** Llega hasta 5.26 °C con un rango inicial de 10–20 °C: excursiones de 2.9–4.7 °C en celdas de al menos 1 m, en el 0.1–0.25 % del volumen. MITgcm (OS7MP) llega a ≤ 0.41 °C en el 0.04 %, y MOM6 z\* no crea ninguno. pyOCEAN σ crea 1.6 °C con ν = 1.

     WENO5-Z no garantiza que T se quede en el rango. En el lock exchange no se nota, pero en la caída por el talud sí. Por eso la RPE de pyOCEAN z\* en el overflow se compara con cautela: el volumen afectado es pequeño, pero va contra el criterio del spec (§A5).
   - MOM6 z\*: **(c)/(d)**, sin explicar; no afecta a pyOCEAN.
9. **Onda interna:**
   - **Amortiguamiento: (a), corregido el 2026-10-14 en z\* y σ.** pyOCEAN amortiguaba la oscilación más que todos los modelos con viscosidad baja: 0.45 con ν = 0.01 y 0.39 con ν = 15, frente a 0.65–0.72 y 0.55–0.65.
     - **Causa:** la velocidad de transporte de las capas, u^θ = θ u^{n+1} + (1 − θ) u^n, aplicaba el θ = 0.6 de la superficie libre también a la parte baroclínica. Con el momento AB3, centrado en n + ½, eso amortigua las ondas internas a ~(θ − ½) ω² Δt.
     - **Diagnóstico**, una variante a la vez, con la energía cinética de la onda en z\* (días 40–60 entre días 0–5): esquema base 0.58; θ = ½ 0.90; θ = 1 0.37; Δt 150 s 0.65 y 600 s 0.50; KE Godunov 0.35; momento en forma de flujo 0.52; trazadores predictor-corrector, continuidad upwind o UP3, sin efecto (0.58). Solo importan θ y Δt.
     - **Corrección:** `theta_bc` (½) pesa la parte baroclínica u′; la media vertical conserva θ, para que los flujos de capa sumen el flujo de la superficie libre. z\* y σ dejan ahora 0.74 / 0.58 / 0.26, en el rango de los demás (0.65–0.72 / 0.55–0.65 / 0.24–0.26). El lock exchange, el seamount y el costo no cambian; el overflow con ν = 1 cambia ~6 % en dRPE/dt.
     - **La híbrida conserva θ_bc = θ** (0.45 / 0.43 / 0.25): con ½, el ruido de sus capas casi vacías deja de amortiguarse y una capa se vacía en el paso lagrangiano (onda interna con ν = 0.01 en el día 103 aun con Δt 150 s; overflow con ν = 1000). θ_bc = 0.55 aguanta, pero z\* dejaría solo 0.57. Pasa a ½ con el limitador de flujo saliente (punto 4 de la lista).
   - **Mezcla:** con la onda sin amortiguar, pyOCEAN z\* mezcla como MITgcm y algo menos que MOM6 z\*. dRPE/dt es 3.14e-4 contra 3.41e-4 y 3.81e-4 con ν = 0.01, y 1.85e-4 contra 1.76e-4 y 2.57e-4 con ν = 15. La versión anterior de este README decía 1.7–2.6 veces menos: era en buena parte efecto de la onda amortiguada.
10. **Seamount:**
    - **pyOCEAN σ con el perfil de referencia (0.22 cm/s)** queda dos órdenes por debajo de CROCO σ (36 cm/s, SM03) y de MOM6 σ (154 cm/s, creciendo). Ninguno de los dos resta un perfil de referencia. La ventaja de pyOCEAN viene de `rho_ref`, que es una opción del modelo. **(c)**, a favor de pyOCEAN.
    - **z\*:** pyOCEAN da 0.08 cm/s y MITgcm 0 exacto.
    - **MOM6 z\* trunca la velocidad desde el día 1, partiendo del reposo con isopicnas planas: sin explicar.** Más que una necesidad de disipación, apunta a un problema de configuración.
      - Ya se probó: Δt 300 s, y `BOUND_CORIOLIS` + `CORIOLIS_EN_DIS` + `MIN_THICKNESS` 0.1.
      - Falta probar las opciones numéricas del ejemplo `seamount` de MOM6: `DTBT = 10`, `BOUND_BT_CORRECTION`, `KH_VEL_SCALE`.

      Mientras no se aclare, el número de MOM6 σ tampoco se toma como su error de gradiente de presión. **Pendiente.**
11. **Costo, al Δt máximo y con 1 proceso en todos:**
    - **Frente a MOM6 y MITgcm:** pyOCEAN z\* es más barato en el lock exchange (0.045 contra 0.076 y 0.106 s/h), en el overflow (0.40 contra 4.3 y 4.7) y en la onda interna (0.0017 contra 0.0025 y 0.0038).
    - **Overflow:** pyOCEAN admite el Δt más grande, 80 s, frente a 20 s de MOM6 y 10 s de CROCO y MITgcm.
    - **Frente a CROCO:** pyOCEAN cuesta 3 veces más en el lock exchange, 2.1 en la onda interna y 1.7 en el seamount, y la mitad en el overflow.
    - **Seamount:** pyOCEAN cuesta 7.6 veces lo de MITgcm (0.20 contra 0.027 s/h). Su Δt máximo es 600 s, la mitad que MOM6 σ y MITgcm (1200 s) e igual que CROCO.
    - **Al mismo Δt de resultados,** CROCO es 3–5.5 veces más rápido que pyOCEAN con 1 hilo en 2-D. Con 5 hilos pyOCEAN no gana en estas mallas.

    - **HYCOM, en el overflow:** con 1 proceso es el más caro, 5.75 s/h con Δt 20 s, 14 veces pyOCEAN z\*. Se midió con ν = 1000, que solo puede favorecerlo. Con 5 procesos baja a 2.84 s/h. Con ν = 100 también falla, pero por una causa sin diagnosticar, en capas casi todas z: no es evidencia sobre las coordenadas isopicnas.

    Lo que pierde pyOCEAN es costo fijo por paso: rendimiento, sin física faltante.
12. **El diagnóstico de RPE de pyOCEAN** (`reference_potential_energy`) interpola el momento linealmente en volumen. Con una hipsometría gruesa se equivoca: con dos puntos en un fondo plano da 1.5 veces el valor exacto. Aquí se evitó refinando la hipsometría a pasos de H/50000. **(a)** en el diagnóstico.

## Qué debe mejorar pyOCEAN

1. ~~**(a) Amortiguamiento de la onda interna con viscosidad baja.**~~ Corregido el 2026-10-14 en z\* y σ (`theta_bc`, punto 9); en la híbrida va con el punto 4.
2. **(a) Extremos nuevos de T con WENO5-Z** en la caída del overflow: hasta 4.7 °C en el 0.25 % del volumen. Opción: un limitador que conserve el rango (Zhang & Shu 2010 [verificar]) o un esquema que preserve la monotonía como opción.
3. **(a) Híbrida, T fuera de rango en capas delgadas** (3.2–33 °C con 5–30 °C). Va al plan de la híbrida, con el lock exchange de 1 h como reproductor. Candidatos de la literatura:
   - Higdon (2005, 2020): atar la velocidad de la capa delgada a la de sus vecinas y poner un piso de grosor en la viscosidad;
   - Hallberg & Adcroft (2009): el reparto del residuo barotrópico.
4. **(a) Híbrida, robustez en el overflow y onda interna:** limitar el flujo saliente de una capa en el paso lagrangiano en lugar de detenerse, como el limitador de celdas que se vacían que ya existe en la forma de flujo (MOM6 limita el flujo a la masa disponible). También las velocidades de 47 m/s con ν = 1000. Con el limitador, la híbrida pasa a θ_bc = ½ y deja de amortiguar la onda interna (0.45 frente a 0.65–0.72). Es el primer punto del plan de la híbrida.
5. **(a) Diagnóstico de RPE:** integrar exactamente ∫ z A dz por tramo en `reference_potential_energy`, para que sea exacto con cualquier hipsometría.
6. **Rendimiento:**
   - reducir el trabajo fijo por paso y elegir el número de hilos según el tamaño de la malla (CROCO es 3–5.5 veces más rápido al mismo Δt en 2-D);
   - revisar por qué el Δt máximo del seamount es la mitad que el de MITgcm y MOM6 σ.

**Para Esteban, fuera de pyOCEAN:**
- Correr CROCO también con WENO5, como en `validation/lock_exchange`, para que su RPE sea comparable.
- Aclarar la configuración de MOM6 z\* en el seamount.
- HYCOM 2.3 se cae en cualquier corrida sin marea (`hnudge`/`salfac` sin reservar, `hycom/README.md`). Aquí se corrigió en la copia del build; vale la pena reportarlo a los desarrolladores de HYCOM.

## Referencias

- Ilicak, M., Adcroft, A. J., Griffies, S. M., Hallberg, R. W. (2012). Spurious dianeutral mixing and the role of momentum closure. *Ocean Modell.* 45–46, 37–58. [10.1016/j.ocemod.2011.10.003](https://doi.org/10.1016/j.ocemod.2011.10.003)
- Petersen, M. R. et al. (2015). Evaluation of the ALE vertical coordinate method in MPAS-Ocean. *Ocean Modell.* 86, 93–113. [10.1016/j.ocemod.2014.12.004](https://doi.org/10.1016/j.ocemod.2014.12.004)
- Chen et al. (2024), *J. Mar. Sci. Eng.* [10.3390/jmse12122249](https://doi.org/10.3390/jmse12122249): confirma el signo de la perturbación de la onda interna.
- Beckmann, A., Haidvogel, D. B. (1993). Numerical simulation of flow around a tall isolated seamount. *J. Phys. Oceanogr.* 23.
- Benjamin, T. B. (1968). Gravity currents and related phenomena. *J. Fluid Mech.* 31.
- Winters, K. B. et al. (1995). Available potential energy and mixing in density-stratified fluids. *J. Fluid Mech.* 289.
- Hallberg, R., Adcroft, A. (2009). *Ocean Modell.* 29. [10.1016/j.ocemod.2009.02.008](https://doi.org/10.1016/j.ocemod.2009.02.008)
- Higdon, R. L. (2005). *J. Comput. Phys.* 206. [10.1016/j.jcp.2004.12.011](https://doi.org/10.1016/j.jcp.2004.12.011)
- Higdon, R. L. (2020). *J. Comput. Phys.* 401. [10.1016/j.jcp.2019.109018](https://doi.org/10.1016/j.jcp.2019.109018)
