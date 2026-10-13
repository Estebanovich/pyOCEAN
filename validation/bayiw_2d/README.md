# BayIW: sección 2-D del eje de la bahía, pyOCEAN contra MITgcm

Spec: `docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md`; plan: `docs/superpowers/plans/2026-10-12-bayiw-2d-section.md`.
Configuración y métricas heredadas del benchmark 3-D (`validation/bayiw_baroclinic/`).

## Qué es

La columna i = 280 de la malla BayIW de MITgcm (352 celdas de 5–13.3 km a lo largo del eje de la bahía, 50 niveles `delR`, 599.9 m), con las entradas de Esteban:
- **Casos:** dos capas (22.77/18.52 °C, interfaz a ≈116 m) y lineal (33.6 → 3.5 °C), con bahía y sin bahía.
- **Forzamiento:** el pulso de viento τ_y de la etapa 1 (8 registros de 1200 s), 60 h.
- **Fronteras:** sur abierta y norte cerrada. Ah = Kh = 100, ν = Kv = 1e-5, f₀ = 6.97e-5, EOS lineal y Δt = 30 s en los dos modelos.
- **pyOCEAN:** z\* (50 capas), híbrida de 25 capas (como el 3-D) y de 50 capas. Todas con 10 capas z ligeras arriba (14.7 m), Flather + Orlanski 0.45 + esponja de 30 celdas sobre u′.
- **MITgcm:** compilado sin MPI con sNx = 3 (`run_mitgcm.sh build`, optfile `darwin_arm64_gfortran_esteban`, sin tocar el código). Cada corrida de 60 h tarda ≈75 s; cada una de pyOCEAN, 0.5–0.7 min.

```bash
bash validation/bayiw_2d/run_mitgcm.sh build
bash validation/bayiw_2d/run_mitgcm.sh run 2layer bay [periodic|nobal|orl]
NUMBA_NUM_THREADS=5 python validation/bayiw_2d/run_pyocean.py 2layer bay --coord hybrid [--layers 50]
python validation/bayiw_2d/compare.py 2layer hybrid orl        # referencia MITgcm: '' | nobal | orl
```

Salidas en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2d/` (`mitgcm/`, `pyocean/`, `compare_*.txt`).

## Decisiones de configuración (rulings)

1. **Paredes en x en los dos modelos.** Con una sola columna, MITgcm es periódico en x: la sección rota (u libre, f activo). pyOCEAN no tiene fronteras periódicas y cierra el canal con paredes (u = 0, f inerte). Por eso MITgcm corre con Nx = 3 y tierra en las columnas 0 y 2. La variante `periodic` (3 columnas de agua) mide lo que la rotación cambia (ver abajo).
2. **Referencia MITgcm "solo Orlanski" (`orl`).** La configuración de Esteban (`data.obcs`) refleja el pulso barotrópico en la frontera sur:
   - la esponja relaja la velocidad *total* hacia 0 (τ = 60 s en el borde), lo que para una onda larga equivale a una pared parcial;
   - `OBCS_balanceFacS = -1` anula el transporte neto por esa frontera (`pkg/obcs/obcs_balance_flow.F:76`), lo que en la sección la cierra para el modo barotrópico.

   Índice de reflexión, rms η(j = 280) 12–27 h / 0–8 h (dos capas con bahía):

   | MITgcm balanceado (Esteban) | sin balance (`nobal`) | solo Orlanski (`orl`) | pyOCEAN Flather |
   |---|---|---|---|
   | 0.57 | 0.41 | 0.02 | 0.02 |

   El pulso reflejado vuelve a la bahía a las 16 h con +3.4 mm. Por eso la comparación se hace contra `orl`. La configuración de Esteban se clasifica **(d)**, un error de configuración del modelo de referencia; puede afectar también sus corridas 3-D.
3. **Muestreo idéntico.** pyOCEAN se remapea de forma conservativa (PPM_H4) a los niveles *fijos* de MITgcm, con η en el nivel superior como su superficie libre lineal, y se muestrea a 110/250 m igual que la salida de MITgcm (`section.to_mitgcm_levels`, `compare.mit_at_depth`). Estirar los niveles con η (como z\*) borraba el desplazamiento barotrópico que MITgcm ve a profundidad fija (efecto de la bahía en Gamma, 1.7e-5 °C). Interpolar entre centros de capas daba pesos de un punto a capas de 1 cm en la interfaz híbrida y convertía la interfaz en una rampa de 200 m. PCM, por su parte, cuantizaba las series dentro de capas gruesas.

## Errores de pyOCEAN encontrados y corregidos (clase a)

1. **Estado inicial híbrido** (`core/dynamics/sigma_baroclinic.py`, `initial_state(..., h_src=...)`).
   - Antes, T se muestreaba en capas σ (24 m arriba en 600 m) y luego se remapeaba. En las columnas profundas los 20 m superiores quedaban uniformes, con un escalón espurio de 0.5 °C a través del talud de la boca.
   - Eso producía 1–4 cm/s desde el reposo, sin viento.
   - Ahora cada pasada de regrid remapea desde la malla de origen, como `Z_INIT_ALE_REMAPPING` de MOM6 (`MOM_state_initialization.F90:2928-2935`).
   - En un canal con escalón en reposo, la velocidad después de 1 h baja de 2.4e-3 a menos de 1e-6 m/s.
2. **Profundidad efectiva de MITgcm** (`validation/bayiw_baroclinic/bayiw.py`, `mitgcm_depth`).
   - MITgcm, con `hFacMin = 1` (`set_defaults.F:180`), redondea el fondo a la interfaz de nivel más cercana (`ini_masks_etc.F:108-120`): la bahía mide 163.3 m (no 164) y la cabeza 126.6 m (no 129.0).
   - `load_inputs` usaba la profundidad cruda, también en el benchmark 3-D.
   - Las celdas parciales llevaban además la T del nivel completo en centros recortados, lo que dejaba saltos de −0.28 °C en el perfil de referencia ρ_ref.
   - Ahora H coincide exacto con `Depth` de la malla de MITgcm.

## Resultados contra MITgcm `orl`

**Seiche Δη = η_bahía − η_sinbahía (0–27 h), estación Seiche:**

| | MITgcm | z\* | híbrida 25 | híbrida 50 |
|---|---|---|---|---|
| Lineal: rango [mm] / correlación | 9.05 / — | 9.18 / 1.00 | 9.20 / 1.00 | 9.19 / 1.00 |
| Dos capas: rango [mm] / correlación | 9.34 / — | 9.48 / 1.00 | 9.60 / 1.00 | 9.49 / 1.00 |

El "período" espectral falla el criterio (3.79 contra 3.66 h en Seiche; 9.0 contra 4.8 h en Seiche2) aunque la correlación es de 1.00. La ventana de 27 h solo tiene un pulso amortiguado de unos 2 ciclos, así que el pico espectral no está resuelto. Es un defecto de la métrica en la sección (c), no del modelo.

**Ondas internas, caso lineal.** Cada celda da amplitud py/MIT y correlación con bahía; estaciones Gamma · C · Omega · Seiche2 · Seiche:

| | T110 | T250 (Gamma, C, Omega) | V110 | Error de la sección de T (6/24/60 h) |
|---|---|---|---|---|
| z\* | 0.97–1.00 / 1.00 | 0.99–1.00 / 1.00 | 0.99–1.01 / 1.00 | 0.06 / 0.11 / 0.04 |
| híbrida 50 | 0.97–1.05 / 1.00 | 1.00 / 1.00 | 0.98–1.00 / 0.99–1.00 | 0.44 / 0.24 / 0.11 |
| híbrida 25 | Gamma–Omega 0.91–1.00 / 1.00; Seiche2 5.6 / 0.66, Seiche 3.1 / 0.74 | 0.96–1.00 / 1.00 | 1.00–1.24 / 0.83–1.00 | 4.9 / 1.6 / 0.8 |

**Ondas internas, dos capas (interfaz a 116 m, T110 a 4.6 m de ella):**

| | T110 (amplitud / corr.) | V110 Gamma · C · Omega · Seiche2 · Seiche (amplitud / corr.) | Espesor de la interfaz MIT/py (60 h) |
|---|---|---|---|
| z\* | 1.02 · 1.30 · 2.26 · 1.59 · 2.33 / 0.91–1.00 | 0.97/0.98 · 0.78/0.97 · 2.19/0.97 · 1.77/−0.36 · 1.43/0.97 | 8.4 / 8.4 m |
| híbrida 25 | 0.01–0.68 (3.5 en Seiche2) / 0.92–1.00 | 0.95/0.92 · 0.61/0.63 · 0.27/0.46 · 0.90/0.99 · 0.93/0.96 | 8.4 / 9.9 m |
| híbrida 50 | 3.2–7.9 / 0.77–0.97 | 11.8/0.00 · 10.7/0.55 · 3.7/0.73 · 1.1/0.52 · 2.3/0.81 | 8.4 / 10.3 m |

**Efecto de la bahía ΔT110 = (T110 − T110₀)_bahía − (T110 − T110₀)_sinbahía**, amplitud py/MIT y correlación en Gamma · C · Omega (criterio del spec: mismo signo y orden de magnitud):

| | Lineal | Dos capas |
|---|---|---|
| z\* | 1.02/0.99 · 0.99/1.00 · 0.99/1.00 | 1.72/0.90 · 1.69/0.91 · 2.21/0.78 |
| híbrida 25 | 1.02/0.99 · 1.03/1.00 · 1.03/0.95 | 0.97/0.99 · 1.87/0.97 · 0.02/−0.02 |
| híbrida 50 | 1.02/0.99 · 1.00/1.00 · 1.00/1.00 | 985/−0.02 · 740/0.02 · 16/−0.22 |

Las señales a 110 m en dos capas son muy pequeñas: el rms de V110 en MITgcm va de 0.06 a 0.67 mm/s y el de T110, de 0.03 °C. La corriente superficial del viento (máx. 0.43 m/s) coincide en todos los casos.

**Salud:** cero truncamientos en las 24 corridas. El contenido de calor cambia menos de 2e-8 en 60 h, con fronteras abiertas.

**Efecto de la rotación (MITgcm periódico contra paredes, los dos con la frontera balanceada):**
- El seiche casi no cambia: correlación 0.96–0.97, rango −4 a −7 %.
- La respuesta baroclínica a 110 m sí cambia mucho. En el caso lineal, la amplitud periódica/paredes es 0.08–0.75 y la correlación −0.32 a −0.09. Es la respuesta de baja frecuencia, con período inercial de 25 h.

## Clasificación

| Diferencia | Clase | Razonamiento |
|---|---|---|
| Período espectral del seiche | (c) métrica | pulso de ~2 ciclos en 27 h; las series tienen correlación 1.00 y rango ±3 % |
| Reflexión en la frontera sur de MITgcm | (d) | esponja sobre la velocidad total + `balanceFacS = -1`; con solo Orlanski, MITgcm = Flather (0.02) |
| Estado inicial híbrido y profundidad de MITgcm | (a), corregidos | ver arriba |
| z\* dos capas: T110 1.3–2.3× y V110 en Omega/Seiche2 cerca de la interfaz | (c) | a 4.6 m de un escalón de T, la serie mide la difusión de la interfaz; MITgcm usa advección centrada de 2.º orden y pyOCEAN WENO5 + remap. En el caso lineal z\* = MITgcm (0.97–1.01) |
| z\* dos capas: 1e-4 m/s desde el reposo en la cabeza de la bahía | (c) | gradiente FV con PLM unilateral en el fondo de la columna somera frente a un escalón de T |
| Híbrida dos capas: T110 casi constante | (c) | la híbrida mantiene la interfaz marcada; en dos capas T110 mide la mezcla numérica, no las ondas |
| Híbrida 25, lineal: T110 en Seiche/Seiche2 y error de la sección | (c) resolución | unas 5 capas isopicnas en 163 m de bahía; con 50 capas todo coincide con MITgcm |
| Híbrida 25, dos capas: V110 en C y Omega (0.27–0.61) | (c)/(b) abierta | boca del talud con pocas capas isopicnas; no diagnosticado |
| Híbrida 25, dos capas: efecto de la bahía en Omega 0.02 | (c)/(b) abierta | misma boca del talud que V110 |
| Híbrida 50, dos capas: ruido de ~1 mm/s a 110 m y efecto de la bahía 16–985× | abierta, posible (a) | velocidad en ~38 capas casi desvanecidas (0–0.1 m) de una interfaz de dos masas de agua; configuración degenerada (50 objetivos para 2 densidades) |
| Rotación (sección periódica) | (b) | pyOCEAN no tiene frontera periódica en x |

## Qué debe mejorar pyOCEAN

1. **Velocidad en capas casi desvanecidas** (híbrida con muchos objetivos vacíos): da ruido de ~1 mm/s. Revisar cómo MOM6/HYCOM acoplan o anulan la velocidad de capas vacías (MOM6 `vertvisc` con `Hmix`/`h_neglect`; HYCOM `momtum` en capas delgadas).
2. **Fronteras periódicas en x/y:** para secciones 2.5-D con rotación y canales.
3. **Gradiente de presión FV en escalones con T discontinua** (1e-4 m/s en z\*): evaluar la ponderación de masa en el fondo (`MASS_WEIGHT_IN_PRESSURE_GRADIENT`) también entre capas no desvanecidas, o PPM en la reconstrucción.
4. **La híbrida de 25 capas en la bahía somera:** más capas isopicnas en zonas someras (objetivos según la profundidad) o una transición σ más amplia.
5. **Métricas:** para la sección, período por cruces por cero o ajuste de un oscilador amortiguado, y en dos capas el desplazamiento de la isoterma central en lugar de T a profundidad fija.
6. **Para Esteban (MITgcm):** su `data.obcs` refleja el pulso barotrópico. Para radiar, usar la esponja solo sobre la parte baroclínica o quitar `balanceFacS = -1`.
