# Onda interna forzada en la frontera: `internal_wave` de MITgcm contra pyOCEAN

- **Spec:** `docs/superpowers/specs/2026-10-12-baroclinic-obc-internal-wave-design.md`
- **Plan:** `docs/superpowers/plans/2026-10-12-baroclinic-obc-internal-wave.md`

## Qué es

`verification/internal_wave` de MITgcm, sin cambiar su código ni sus entradas, corrido 10 períodos (5.2 días) en lugar de 100 pasos.

**Configuración:**
- **Sección x–z:** 60 columnas de 218 m (Lx = 13.1 km) y 20 niveles de 10 m.
- **Fondo:** 200 m, con una pendiente de 0.03 desde x = 2.5 km hasta 40 m de profundidad; la última columna es tierra.
- **Física:** N² = 1e-6, f = 0.
- **Frontera oeste:** su `obcs_calc.F` impone una onda de modo 1 (u₀ = 0.024 m/s, período 44 567 s, media vertical nula) y la T que le corresponde. La pendiente es subcrítica (0.03 frente a ω/N = 0.14), así que la onda sube por ella, se peralta y se disipa.

**pyOCEAN:**
- **Frontera oeste `'clamped'`:** usa el nuevo estado exterior que varía en el tiempo (`open_boundary_state`). La función `case.obcs_west` reproduce `obcs_calc.F` línea por línea.
- **Frontera este:** pared. La frontera de MITgcm cae sobre tierra.
- **Corridas:** z\*, híbrida y σ, con 20 capas, Δt = 500 s (como MITgcm), las mismas viscosidades y difusividades, y la profundidad con celdas parciales de MITgcm (`hFacMin` = 0.2, `case.mitgcm_depth` = `Depth` de su malla).

```bash
bash validation/internal_wave/run_mitgcm.sh build && bash validation/internal_wave/run_mitgcm.sh run
NUMBA_NUM_THREADS=5 python validation/internal_wave/run_pyocean.py zstar [--ke arakawa]
python validation/internal_wave/compare.py zstar[_arakawa]
```

**Costo:** MITgcm tarda ~1 s; pyOCEAN, 0.1 min por corrida. Las salidas van en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/internal_wave/`.

**Geometría:** MITgcm impone T en su celda de frontera (columna 1, `obcs_apply_ts.F:205`) y u en la cara al este de ella (`obcs_apply_uv.F:205`). Por eso el dominio de pyOCEAN son las columnas 2–60 de MITgcm, con la cara oeste en ese punto u. MITgcm guarda en el paso más cercano a cada marca de 1800 s y pyOCEAN en el primero que la pasa; los armónicos se ajustan con los tiempos de cada modelo.

## Prueba analítica previa (solo pyOCEAN, `tests/test_sigma_ale.py`)

**Configuración:** un canal plano de 200 m y 10 km (dx = 100 m), con el oeste `'clamped'` (modo 1) y el este `'radiating'` con una esponja de 2 km.

**Solución:** la onda progresiva u = u₀ cos(πz/H) sin(ωt − kx), con k = πω/(NH).

**Resultado:** en z\* y en la híbrida, la amplitud queda dentro de ±5 % de la onda con su decaimiento viscoso lineal, y la fase dentro de 3.4°.

**Lo que esta prueba encontró y corrigió:**
1. **La esponja en un borde impuesto.** Una banda relajada hacia el perfil del borde, copiado a lo largo de la normal, forzaba 2 km en fase: un error de fase de 160°. Ahora los bordes `'clamped'` no llevan esponja.
2. **El exterior de un borde sin datos.** Con datos solo en el oeste, la esponja del este se relajaba hacia el perfil del *oeste*. Ahora cada punto toma el exterior de su borde abierto más cercano, y los bordes sin datos mantienen el exterior en reposo.
3. **La energía cinética Godunov amortigua la onda progresiva.** Por defecto, el modelo pierde ~9 % de amplitud *extra* en 5 km con dx = 100 m (8 % con dx = 50 m: es de primer orden en dx).
   - Con Arakawa o con el momento en forma de flujo desaparece.
   - El flujo de grosor (`upwind` o PPM) y la advección de T no influyen.
   - El test usa Arakawa para probar la frontera y no el esquema de momento.

## Resultados contra MITgcm (últimos 4 períodos)

**Modo 1 en la parte plana (x < 2.5 km).** Proyección de u sobre cos(πz/H) y de T′ sobre sin(πz/H). Criterio: ±10 % y ±10°.

| | u: amplitud py/MIT · fase | T′: amplitud · fase |
|---|---|---|
| z\* (Godunov, por defecto) | 0.84–0.99 · −2 a −9° ❌ | 0.94–1.07 · −1 a −9° ✅ |
| híbrida (Godunov) | 0.83–0.99 · −2 a −6° ❌ | 0.96–1.04 · 0 a −5° ✅ |
| σ (Godunov) | 0.84–0.98 · −2 a −8° ❌ | 0.98–1.05 · −1 a −8° ✅ |
| z\* (Arakawa) | 0.93–1.01 · −2 a −7° ✅ | 0.98–1.08 · −2 a −8° ✅ |
| híbrida (Arakawa) | 0.91–1.01 · +1 a −4° ✅ | 0.98–1.07 · +1 a −3° ✅ |
| σ (Arakawa) | 0.92–0.99 · −2 a −7° ✅ | 0.99–1.10 · −2 a −8° ❌ (1.10 en el límite) |

**Flujo de energía baroclínica ⟨∫(u − ū)(p′ − p̄′) dz⟩ [W/m].** p′ se calcula a partir de T y η igual en los dos modelos.

| x [km] (H [m]) | 1 (200) | 2 (200) | 3 (177) | 4 (144) | 4.5 (132) | 5 (118) | 6 (85) | x del 50 % [km] |
|---|---|---|---|---|---|---|---|---|
| MITgcm | 1.60 | 1.52 | 1.56 | 1.42 | 0.84 | 0.52 | 0.06 | 4.61 |
| z\* | 1.52 | 1.40 | 1.27 | 0.95 | 0.75 | 0.49 | 0.08 | 4.53 ✅ |
| híbrida | 1.53 | 1.43 | 1.25 | 0.95 | 0.77 | 0.48 | 0.05 | 4.53 ✅ |
| σ | 1.54 | 1.41 | 1.28 | 1.10 | 0.92 | 0.62 | 0.11 | 4.74 ✅ |
| z\* Arakawa | 1.63 | 1.58 | 1.55 | 1.51 | 1.31 | 0.71 | 0.17 | 4.83 ✅ |
| híbrida Arakawa | 1.64 | 1.56 | 1.50 | 1.50 | 1.16 | 0.55 | 0.01 | 4.67 ✅ |
| σ Arakawa | 1.64 | 1.56 | 1.48 | 1.68 | 1.39 | 0.73 | 0.30 | 4.85 ✅ |

(u centrado en las celdas con las caras bajo el fondo = 0 en los dos modelos: MITgcm escribe 0 ahí y el remapeo de pyOCEAN daba NaN; la versión anterior de esta tabla sesgaba la pendiente 6–10 % contra pyOCEAN.)

- **A 2 km**, el flujo de pyOCEAN es 0.92–0.94 del de MITgcm con Godunov y 1.02–1.04 con Arakawa (criterio ±15 %).
- **En la pendiente**, la onda se peralta y su escala vertical baja como H, así que la disipación crece. Los dos modelos pierden el flujo en la misma zona: el 50 % cae a 4.53–4.85 km en pyOCEAN y a 4.61 km en MITgcm.
- **Más allá de 6 km** el flujo es una cola de menos del 5 %, y su cociente no es significativo.

**2.º armónico** cerca del fondo en la zona somera: 0.08 contra 0.03 mm/s a 9 km, despreciable en los dos.

**Salud:** cero truncamientos; max|Σh − H − η| ≤ 1e-13 m. El contenido de T cambia por el flujo en el borde abierto (T ≈ ±0.05 °C alrededor de 0).

## Clasificación

| Diferencia | Clase | Razonamiento |
|---|---|---|
| u de modo 1 con Godunov: 0.83–0.99 en 2.5 km | (c) | El esquema de energía cinética Godunov (por defecto, elegido para corrientes de densidad) amortigua las ondas internas progresivas, con error de primer orden en dx. Con Arakawa, pyOCEAN = MITgcm (0.91–1.01) |
| Flujo en la pendiente (forma del decaimiento entre 3 y 6 km) | (c) | La disipación de la onda que se peralta depende de la resolución vertical en aguas someras y de la disipación numérica; la posición del 50 % coincide a ±0.24 km |
| σ Arakawa: T′ 1.10 en un punto | (c) | En el límite del criterio; σ comprime capas sobre la pendiente |
| Borde `'clamped'`: pyOCEAN deja el modo barotrópico con Flather; MITgcm fija el transporte total (0) | (c) | El forzamiento tiene media vertical nula; los dos coinciden en la parte plana |

No se encontró ningún error de clase (a) en la comparación con MITgcm. Los dos errores de la frontera (esponja en el borde impuesto, exterior del borde sin datos) salieron en la prueba analítica y ya están corregidos.

## Qué debe mejorar pyOCEAN

1. **La energía cinética Godunov como valor por defecto amortigua las ondas internas.** Con dx = 100 m pierde ~9 % de amplitud en 5 km; en la sección es el 15 % de u en la parte plana. Para estudios de ondas internas (etapa E, la tesis de Esteban) conviene Arakawa, o un esquema por defecto según el régimen. Es una decisión para Esteban.
2. **La esponja con un exterior que varía en el tiempo** usa el perfil del borde copiado a lo largo de la normal, como MITgcm OBCS. Para ondas que entran por un borde `'radiating'` eso fuerza la banda en fase. Para anidar en HYCOM hará falta un exterior 3-D en toda la banda (la siguiente pieza de `OBData`).
3. **Frontera de transporte total** en el modo `'clamped'` (el equivalente exacto de MITgcm), si algún caso lo necesita.
