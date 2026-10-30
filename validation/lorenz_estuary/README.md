# Benchmark: estuario idealizado de Lorenz et al. (2021)

- **Spec:** `docs/superpowers/specs/2026-10-28-n2a-salinity-freshwater-design.md` §6
- **Plan:** `docs/superpowers/plans/2026-10-28-n2a-salinity-freshwater.md`, Task 8
- **Artículo:** Lorenz, Klingbeil & Burchard (2021), *Impact of evaporation and precipitation on estuarine mixing*, JPO 51, 1319–1333, doi:10.1175/jpo-d-20-0158.1. Copia local: `Lorenz_2021 - Impact of Evaporation and Precipitation on Estuarine Mixing.md` (§4b l. 838-857, Tabla 1 l. 881-906, pie de la Fig. 2 l. 827-837, Tabla 2 l. 923-1030); la posición de los transectos se leyó de la Fig. 2 del PDF (p. 1324).

## Qué es

Un canal 2-D con evaporación, precipitación, río y marea, donde la mezcla total del modelo dentro del estuario debe ser **exactamente** la de la relación (32) del artículo, evaluada con TEF en el transecto. GETM lo cumple "down to machine accuracy" (Tabla 2). En pyOCEAN prueba juntos la salinidad, el agua dulce real, el balance de varianza discreto (Task 6) y el TEF dentro del modelo (Task 7). Después se comparan las magnitudes con la Tabla 2 de GETM.

## Configuración

| | Valor | Fuente |
|---|---|---|
| Canal | 100 km, dx = dy = 200 m, 500 × 1 celdas | §4b |
| Fondo | lineal de 15 m (océano, oeste) a 5 m (río, este); H en los centros de celda | §4b |
| Vertical | 60 capas σ uniformes | §4b |
| Frontera oceánica | S = 30 g/kg; M2 de 0.5 m en B–D como **elevación prescrita**: η de la celda oceánica = a sin(ωt) (modo `'eta'`, Dirichlet, como GETM y CROCO `OBC_COM_ZSPECIFIED`, `OCEAN/zetabc.F:95-99`); el flujo de la cara de borde sale de la continuidad de esa celda, así que el volumen se conserva exactamente; u′ y S como `'radiating'` (entrada S = 30 + esponja) | §4b l. 843-845 (decisión R4); fase inicial 0 sin rampa: **[no dado en el artículo]** |
| Agua dulce | río, E integrada y P integrada: 5.0 m³/s cada uno (Tabla 1) | §4b, Tabla 1 |
| Zonas | A: E en la mitad interior (x ≥ 50 km). B–D: E en el tercio medio (celdas 167–332), P en el tercio este (333–499) | pie de la Fig. 2 |
| Transectos | borde oeste de la zona de evaporación: cara 250 (x = 50 km) en A, cara 167 (x = 33.4 km) en B–D; la región es todo lo que está al este | líneas discontinuas de la Fig. 2 |
| T | constante (10 °C), α = 0 | §4b ("temperature is kept constant"); el valor: **[no dado en el artículo]** |
| Integración | 500 ciclos M2 hasta el estado cuasi periódico + promedio de 100 ciclos (600 en total); en A, sin marea, el ciclo M2 es solo la unidad de tiempo | §4b l. 851-857 |
| TEF | 250 bins de 0 a 40 g/kg, cada paso, con los flujos efectivos de la advección | §4b l. 846-849 |

| Tabla 1 | Q_r | Q_evap | Q_precip | M2 |
|---|---|---|---|---|
| A | 0 | −5 | 0 | 0 |
| B | 5 | −5 | 0 | 0.5 m |
| C | 0 | −5 | 5 | 0.5 m |
| D | 5 | −5 | 5 | 0.5 m |

**Parámetros que el artículo no da** (ninguno ajustado a la Tabla 2):

| Parámetro | Valor | Elección |
|---|---|---|
| ρ₀ | 1025 kg/m³ | **[no dado en el artículo]**; `rho_0` de GETM [verificar] |
| β (EOS lineal) | 0.78 kg m⁻³ (g/kg)⁻¹, S₀ = 30 | **[no dado en el artículo]**; `dsr0` de la EOS lineal de GETM [verificar]. GETM puede usar UNESCO por defecto (β ≈ 0.77 a S = 30, T = 10 °C) |
| κ_v de S y ν (ALE-σ) | 1e-4 m²/s constante | **[no dado en el artículo]**; GETM usa k-ε de GOTM para todo y fondo `avmback = avhback = 0` [verificar]; 1e-4 es el `nu_const` por defecto de pyOCEAN, la constante más simple |
| κ_v de S (σ + GLS) | κ_t del k-ε estratificado, piso 1e-6 m²/s (`kv` = 0, Task 8c) | el fondo de CROCO con GLS (`nuws`, gls_mixing.F:152); el más cercano al fondo cero de GETM [verificar] |
| ν_h | 0 | **[no dado en el artículo]**; `Am = 0` de GETM [verificar] |
| κ_h | 0 | lo exige la identidad exacta (el flujo difusivo en el transecto no tiene volumen y TEF no lo puede clasificar) |
| Rugosidad del fondo | z0 = 1e-3 m (σ + GLS y ALE-σ + GLS, ley log en la capa de fondo); C_d = [κ/(ln(D/z0) − 1)]² = 2.37e-3 con D = 10 m sobre la velocidad media de la columna (ALE-σ, `hbbl` = 20 m) | **[no dado en el artículo]**; el mismo z0 en las tres configuraciones (derivado, no ajustado) |
| Esponja oeste | 10 celdas (2 km), perfil MITgcm, τ = (1200, 60) s | **[no dado en el artículo]**; GETM impone S en la frontera sin esponja; su borde interior (x = 2 km) queda a 31.4 km del transecto de B–D (x = 33.4 km) y a 48 km del de A |
| Estado inicial | reposo, η = 0, S = 30 en todo el canal | **[no dado en el artículo]** |
| f | 0 (corte 2-D de una fila) | **[no dado en el artículo]** |
| Δt | T_M2/746 ≈ 59.9 s (`sigma_gls`) y T_M2/498 ≈ 89.8 s (`ale_sigma`), número entero de pasos por ciclo | búsqueda abajo (σ + GLS bajó de T_M2/373 tras las corridas largas con el GLS estratificado) |

**Configuraciones de pyOCEAN** (spec §6, Esteban; `ale_sigma_gls` de N3 Task 10, criterio 4 del spec N3):

| | `sigma_gls` | `ale_sigma` | `ale_sigma_gls` |
|---|---|---|---|
| Coordenada | σ euleriana | ALE con regla σ, PGF de volumen finito | ALE con regla σ, PGF de volumen finito |
| Cierre | GLS k-ε estratificado (Canuto A con flotabilidad, Task 8c): ν para el momento y κ_t para S | ν y κ constantes | el GLS de `sigma_gls` en ALE (N3 Task 5), `kv` = 0 |
| Fondo | ley logarítmica, z0 = 1e-3 m | arrastre cuadrático, C_d = 2.37e-3 | ley logarítmica, z0 = 1e-3 m, sobre la capa de fondo d_b = max(h_b, min(`hbbl`, D)) con `hbbl` = 0.05 m |
| Trazadores | WENO5, SSP-RK3 | WENO5 + FCT, SSP-RK3, remapeo PPM_H4 | como `ale_sigma` |

**`hbbl` de `ale_sigma_gls`.** La ley de pared de ALE (N3 Task 4, `core/components/bottom_drag.py`) actúa sobre una capa de fondo de grosor d_b = max(h_b, min(`hbbl`, D)). Con el `hbbl` = 20 m de `ale_sigma`, toda la columna sería la capa de fondo. Con `hbbl` ≤ h_b en todas partes, d_b = h_b y la ley es la de σ (C_d de la primera capa sola, `test_ale_loglaw_matches_sigma_column`). h_b = D/60 vale 0.083 m en el extremo del río (D = 5 m) y la marea más baja deja D ≥ 4.5 m; 0.05 m queda por debajo de h_b hasta D = 3 m.

**Diferencias ya conocidas antes de correr:**
- **GLS con flotabilidad y sobre los trazadores (resuelto, Task 8c; reemplaza el Review Focus 5 de N2a).** Antes, en `sigma_gls` S se difundía con la constante `kv` y el GLS no tenía flotabilidad (N² = 0): con Ri ≈ 80 el cierre daba ν ≈ 1e-3–1e-2 m²/s y frenaba la circulación de intercambio; A σ+GLS no llegaba a un estado cuasi estacionario (S_max 191 g/kg). Ahora el k-ε es el de CROCO `gls_mixing.F` (N² de ρ(S), B = −κ_t N², β₃⁻ = −0.4, α_N en Canuto A, límite de Galperin) y su κ_t difunde S, como el k-ε de GETM (`tests/test_lorenz_estuary.py::test_gls_config_uses_gls_tracer_diffusivity`; `tests/test_gls_stratified.py`). Smoke de 50 ciclos de A abajo.
- **Advección (c).** GETM usa TVD Superbee (Pietrzak 1998); pyOCEAN, WENO5 (más FCT en ALE).
- **Marea en la frontera (resuelto, R4).** Con Flather hacia η_ext = 0.5 sin ωt y ū_ext = 0 (la primera versión), la celda oceánica tenía 0.16 m de amplitud y la cabeza 0.50 m: el canal mide ≈ 0.225 λ_M2, cerca del cuarto de onda, y la boca queda casi en un nodo. Con la elevación prescrita (modo `'eta'`), la celda oceánica tiene 0.500 m y la cabeza 1.03 m (ALE-σ) y 1.07 m (σ + GLS) en B, en el cuarto ciclo; las corrientes llegan a 0.8 m/s.

**Supuestos de la identidad exacta** (`case.check_identity_assumptions`, se verifican al construir): κ_h = 0; ninguna celda de esponja en la región; toda cara abierta de la región es un río `'clamped'` de agua dulce (S = 0) que solo entra; salinidad, balance de varianza y TEF activos.

## Búsqueda de Δt (5 hilos, Δt = T_M2/N, marea con elevación prescrita)

Estable = campos finitos, S en un rango físico y sin truncamientos CFL en ALE, durante 4–10 ciclos M2 desde el estado inicial.

| Config | Exp. | Δt [s] (N) | Resultado |
|---|---|---|---|
| `sigma_gls` | A | 452 (99), 361 (124), 240 (186) | inestable a las 22–25 h: la advección vertical explícita de S en la capa superior (≈ 0.13 m) bajo el chorro superficial de evaporación (u ≈ 0.09 m/s en x ≈ 60 km) |
| `sigma_gls` | A | 120 (373) | estable 10 ciclos |
| `ale_sigma` | A | 90 (498) | estable 10 ciclos |
| `sigma_gls` | B | 240 (186) | inestable a las 23 h |
| `sigma_gls` | B | 180 (248), 120 (373) | estables 10 ciclos |
| `sigma_gls` | D | 120 (373) | estable 10 ciclos |
| `ale_sigma` | B | 240 (186), 180 (248), 150 (298) | se detiene por truncamientos CFL (> 1000) a las 6.5, 7.1 y 19.8 h: la vaciante de 0.67 m/s en la cara de la boca, \|u\| Δt/Δx > 0.5 (`truncate_cfl`, MOM6 `vertvisc_limit_vel`) |
| `ale_sigma` | B, D | 120 (373) | estables 10 ciclos, sin truncamientos; pico de 0.78 m/s, \|u\| Δt/Δx = 0.47 |

La búsqueda de arriba se hizo con el GLS sin flotabilidad (antes de Task 8c) y dio T_M2/373 ≈ 119.9 s para `sigma_gls`. **Con el GLS estratificado eso no basta en las corridas largas** (`*_sigma_gls_dt120`, guardadas como contraste):

| Config | Exp. | Δt [s] (N) | Resultado (600 ciclos) |
|---|---|---|---|
| `sigma_gls` (8c) | A | 120 (373) | completa: S_reg 32.8, S_max 36.9, Me 294.6 |
| `sigma_gls` (8c) | B, C, D | 120 (373) | revientan en los ciclos 126, 134 y 99 |
| `sigma_gls` (8c) | A–D | 60 (746) | las cuatro completas; \|u\| Δt/Δx ≤ 0.36 en A–C y 0.46 en la intrusión de D (registros cada ¼ de ciclo) |

Diagnóstico (repetición desde el checkpoint): el GLS estratificado deja ν ≈ 1e-5 m²/s bajo la haloclina, así que en la vaciante se forma un chorro superficial de 1–1.5 m/s cerca de x ≈ 68 km (el borde este de la zona de evaporación). Ahí \|u\| Δt/Δx = 0.75–0.9, más allá del límite de la advección de momento en forma de flujo AB3 + WENO5 con A_h = 0. Un modo de 3–4 Δx crece ×1.3 por paso, y luego la matriz de la superficie libre se vuelve singular (`spsolve`: "Matrix is exactly singular"). En σ no hay un respaldo como el `truncate_cfl` de ALE. Con T_M2/746 ≈ 59.9 s el número de Courant se reduce a la mitad: ≤ 0.36 en A–C y 0.46 en el pico de la intrusión de D. Es una elección numérica (c): no cambia la física, y A, que completó a 120 s, da Me 294.6 contra 270.2 a 60 s (9 %, dentro de la variabilidad de A; ver Estacionariedad).

**`ale_sigma_gls` (N3 Task 10a):**

| Config | Exp. | Δt [s] (N) | Resultado (150 ciclos) |
|---|---|---|---|
| `ale_sigma_gls` | D | 89.8 (498) | se detiene en el ciclo 131 por truncamientos CFL (> 1000): el chorro superficial de vaciante, u = −1.17 m/s en la capa 57 de la cara i = 259 (\|u\| Δt/Δx = 0.53); \|u\|max registrado 1.10 m/s |
| `ale_sigma_gls` | D | 59.9 (746) | completa: \|u\|max 1.03 m/s (\|u\| Δt/Δx ≤ 0.31), n_limited 0, \|Me − M\|/M 1.4e-10 |

Es el mismo chorro de 1–1.5 m/s de `sigma_gls` (el GLS deja ν ≈ 1e-5 bajo la haloclina); en ALE lo detiene `truncate_cfl` en vez de reventar. Se usa T_M2/746, como en `sigma_gls` (c).

**`sigma_gls` + Smagorinsky (spec 2026-10-30, criterio 7):** D, Δt = T_M2/498 ≈ 89.8 s, C = 0.15, 150 ciclos, promedio de los últimos 50. Se comparó con los primeros 150 ciclos de la corrida de producción de T_M2/746 (`D_sigma_gls`, serie de ¼ de ciclo).

```bash
python validation/lorenz_estuary/run_pyocean.py D sigma_gls --smag 0.15 --dt 89.78747855421688 --cycles 150 --avg 50
```

Salida: `D_sigma_gls_smag0.15/`. Bitácora: `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/smag_logs/lorenz_D_smag0.15.log`.

| | T_M2/746, sin Smagorinsky | T_M2/498, C = 0.15 |
|---|---|---|
| completa 150 ciclos | sí | sí |
| \|u\|max (registros de ¼ de ciclo), ciclos 0–50 / 50–100 / 100–150 | 0.89 / 1.04 / 1.11 m/s | 0.81 / 1.14 / **2.06** m/s |
| \|u\| Δt/Δx máximo | 0.33 | **0.92** (ciclo 120.8) |
| media de \|u\|max, ciclos 100–150 | 0.761 m/s | 0.782 m/s |
| S_max, ciclos 100–150 | 38.79 | 38.96 |
| S_reg medio, ciclos 100–150 / en el ciclo 150 | 20.57 / 16.96 | 20.55 / 16.73 |
| n_limited / truncamientos | 0 / — (σ) | 0 / — (σ) |
| \|Me − M\|/M (ventana de los ciclos 100–150) | — | 8.5e-12 ✅ (≤ 5e-10) |
| Me (ciclos 100–150) | no disponible | 3782.65 |
| s por paso / s por ciclo M2 | 0.0137 / 10.2 | 0.0145 / 7.2 |

- **Sí completa, pero sin margen.** Hay tres ráfagas, en los ciclos 86–94, 118–122 y 144–148. En ellas \|u\|max llega a 1.14, 2.06 y 1.54 m/s (\|u\| Δt/Δx = 0.51, 0.92 y 0.69). El ν de Smagorinsky llega a su cota de estabilidad (`nu_t_max` = 24.3 m²/s = 0.8 · (6/11) / (4 Δt (2/Δx²))). `n_visc_limited` (acumulado) salta solo en esas ráfagas: 0 → 11 417 → 17 047 → 28 554.
  - Como esta es una malla σ sin capas evanescentes, cada punto limitado es un punto donde C Δ² |S| pasó de la cota.
  - La corrida de T_M2/746 nunca pasa de 1.11 m/s.
  - Los registros son cada ¼ de ciclo, así que entre ellos las ráfagas pudieron ser mayores.
- **El mismo chorro de vaciante.** Las ráfagas caen en la misma época que el chorro de vaciante que detuvo `ale_sigma_gls` a T_M2/498 en el ciclo 131, y que hizo reventar `sigma_gls` a T_M2/373 en el ciclo 99 (arriba). Smagorinsky lo frena lo bastante para terminar 150 ciclos, pero no lo quita.
- **Falta el control sin Smagorinsky a T_M2/498 en σ + GLS** (no se corrió: 150 ciclos ≈ 18 min). Por eso no está probado que Smagorinsky sea lo que deja terminar la corrida. Lo que sí está probado es que con Smagorinsky termina.
- **Me no se puede comparar en la misma ventana.** La corrida de T_M2/746 solo tiene los acumuladores de TEF de los ciclos 500–600 (Me 800.7). S_max queda a 0.4 % y S_reg a 0.1 % (media de los ciclos 100–150) y 1.4 % (ciclo 150) de la corrida de T_M2/746.
- **La relación exacta de Lorenz sigue cerrando** (8.5e-12): la viscosidad solo toca el momento. ✅
- **Decisión:** se sigue usando **T_M2/746** en producción para `sigma_gls`. Pasar a T_M2/498 (30 % más barato por ciclo) pediría el control sin Smagorinsky y una corrida de 600 ciclos sin ráfagas que pasen de \|u\| Δt/Δx ≈ 0.5. Clase (c), elección numérica.

Se usa **Δt = T_M2/746 ≈ 59.9 s en `sigma_gls` y `ale_sigma_gls`** y **T_M2/498 ≈ 89.8 s en `ale_sigma`**: 120 s ya deja \|u\| Δt/Δx = 0.47 junto al umbral de truncamiento de 0.5. El runner revisa que los campos sean finitos cada ¼ de ciclo y antes de cada checkpoint, y guarda uno cada 25 ciclos.

## Costo (2 h simuladas, 5 hilos, TEF y balance de varianza activos; 600 ciclos M2)

| Corrida | s/paso | Pasos | Reloj extrapolado |
|---|---|---|---|
| A `sigma_gls` | 0.051 | 223 800 | 3.2 h |
| A `ale_sigma` | 0.018 | 298 800 | 1.5 h |
| B `sigma_gls` | 0.051 | 223 800 | 3.1 h |
| B `ale_sigma` | 0.020 | 298 800 | 1.7 h |
| C, D | como B | | ≈ 3.1 h (σ) y ≈ 1.7 h (ALE-σ) |
| **Total de las 8, una a la vez** | | | **≈ 19 h** |

Medido con la máquina cargada (promedio de carga 17–20) y antes de fijar los hilos de BLAS, así que es una cota superior.

**Producción (600 ciclos, 5 hilos, `OPENBLAS_NUM_THREADS` = `OMP_NUM_THREADS` = `VECLIB_MAXIMUM_THREADS` = 1):**

| Config | Δt | Pasos | s/paso | Reloj por corrida |
|---|---|---|---|---|
| `sigma_gls` | T_M2/746 ≈ 59.9 s | 447 600 | 0.0133–0.0137 | 1.65–1.71 h |
| `ale_sigma` | T_M2/498 ≈ 89.8 s | 298 800 | 0.0153–0.0160 | 1.27–1.33 h |

Las 8 corridas suman ≈ 12 h.

`ale_sigma_gls` (N3, 2026-10-29/30, Δt = T_M2/746, 447 600 pasos): 0.0198–0.0217 s/paso, 2.47–2.70 h por corrida; las cuatro ≈ 10.3 h. Cuesta 1.5–1.6 × `sigma_gls` y 1.9–2.1 × `ale_sigma` por corrida (el GLS en ALE y el Δt de σ + GLS).

`ale_sigma_gls` (N3 Task 10a, 2 h simuladas, Δt = T_M2/746, 447 600 pasos por corrida): A 0.0186 s/paso → 2.3 h; B 0.0194 → 2.4 h; D en la búsqueda de Δt, 150 ciclos: 0.020 → 2.5 h. Las cuatro ≈ 10 h, una a la vez. Al principio, el pool de hilos de OpenBLAS giraba junto a los 5 hilos de Numba y la corrida iba 4× más lenta (`progress.md`, 2026-10-28).

## Cómo reproducirlo

```bash
python validation/lorenz_estuary/run_pyocean.py B sigma_gls --hours 2           # costo: 2 h simuladas, no escribe nada
python validation/lorenz_estuary/run_pyocean.py B sigma_gls                     # 600 ciclos, promedio de los últimos 100
python validation/lorenz_estuary/run_pyocean.py B sigma_gls --resume            # sigue desde el último checkpoint
python validation/lorenz_estuary/run_pyocean.py B ale_sigma_gls                 # ALE-σ + GLS (N3 Task 10)
python validation/lorenz_estuary/compare.py                                     # Tabla 2: GETM contra pyOCEAN
```

Las salidas van a `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/lorenz_estuary/<exp>_<config>/`, nunca al repositorio: `summary.json`, `section.npz`, `series.npz` y el checkpoint (`restart.nc` + `restart_extra.pkl`, con los acumuladores de TEF que `core/restart.py` no guarda). Reanudar reproduce bit a bit la corrida sin interrumpir (verificado en B, las dos configuraciones, 0.5 ciclos); `--resume` se niega si `--cycles`, `--avg` o Δt no son los del checkpoint.

## Smoke de A σ + GLS antes y después de Task 8c (50 ciclos M2, Δt = T_M2/373, 2 hilos)

| | antes de 8c (`gls_stratified=False`, `kv` = 1e-4) | después (`gls_stratified=True`, `kv` = 0) |
|---|---|---|
| mediana de ν bajo la zona de evaporación, ciclo 50 | 2.4e-3 m²/s | 2.0e-3 m²/s (κ_t: 2.7e-3 m²/s) |
| \|u\|max, ciclo 50 | 0.031 m/s | 0.095 m/s |
| pendiente de S_reg, ciclos 40–50 | +0.089 por ciclo | +0.041 por ciclo (ciclos 1–10: +0.087) |
| S_max, ciclo 50 | 37.6 g/kg | 37.5 g/kg |
| costo por paso (mediana de 20) | 0.0185 s | 0.0195 s (+5 %) |

En A, la evaporación hace más salada el agua de superficie, así que la columna queda inestable (N² < 0). Con flotabilidad, el k-ε convecciona y mezcla S (κ_t ~ 3e-3 m²/s). La circulación de intercambio se triplica y la sal se acumula a la mitad de ritmo. Falta ver si llega a un estado cuasi estacionario: eso requiere la corrida de 600 ciclos.

## Resultados

Corridas de producción del 2026-10-28/29: 600 ciclos M2 desde el reposo con S = 30 y promedio de los últimos 100, `sigma_gls` con el GLS estratificado de Task 8c a Δt = T_M2/746 y `ale_sigma` a T_M2/498. Salidas en `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/lorenz_estuary/<exp>_<config>/`. Las tablas salen de `python validation/lorenz_estuary/compare.py`, que también imprime la razón de cada clase. Ninguna corrida tuvo `n_limited` > 0 ni truncamientos. `ale_sigma_gls` (N3 Task 10, criterio 4 del spec N3) se corrió el 2026-10-29/30 a T_M2/746, con el mismo arranque; tampoco tuvo `n_limited` ni truncamientos.

### Tabla 2: GETM contra pyOCEAN

Clase de cada fila (CLAUDE.md, Validation Strategy): (a) error nuestro, (b) física que falta, (c) elección numérica o de configuración, (d) error del modelo de referencia; "?" = hipótesis. "—" = coincide o es exacta.

**Experimento A** (Q_r, Q_evap, Q_precip, M2 = (0.0, -5.0, 0.0, 0.0))

| Cantidad | GETM | pyOCEAN sigma_gls | pyOCEAN ale_sigma | pyOCEAN ale_sigma_gls | Clase |
|---|---|---|---|---|---|
| s_in [g/kg] | 30 | 30.01 | 31.19 | 30.01 | c |
| s_out [g/kg] | 32.67 | 33.1 | 32.59 | 33.33 | c |
| [(s²)_in]^½ [g/kg] | 30 | 30.01 | 31.19 | 30.01 | c |
| [(s²)_out]^½ [g/kg] | 32.73 | 33.15 | 32.59 | 33.39 | c |
| dV [m³/s] | 0 | 0.01015 | -1.904e-09 | -0.01296 | — |
| dS [g/kg m³/s] | 1.18 | 4.482 | -6.104e-08 | -10.93 | c |
| dS2 [(g/kg)² m³/s] | 65.3 | 316.3 | -1.958e-06 | -751.7 | c |
| F_surf^{s²} [(g/kg)² m³/s] | 5279 | 5440 | 5714 | 5388 | c |
| Me | 212.3 | 270.2 | 631.7 | 281 | c |
| Mc | 403.9 | 428.8 | 632.1 | 458.1 | c |
| Mp | 203.6 | 309.1 | 631.7 | 220 | c |
| Mcp | 396.7 | 472.3 | 632.1 | 385.5 | c |
| M_model^tot | 212.3 | 270.2 | 631.7 | 281 | c |
| M_model^phy | 103.5 | 248.2 | 631.5 | 263.6 | c |
| M_model^num | 108.8 | 22 | 0.1613 | 17.39 | c |
| \|Me − M_tot\|/M_tot | precisión de máquina | 7.8e-11 | 4.3e-10 | 1.5e-09 | c (> 5e-10, remap round-off) |

**Experimento B** (Q_r, Q_evap, Q_precip, M2 = (5.0, -5.0, 0.0, 0.5))

| Cantidad | GETM | pyOCEAN sigma_gls | pyOCEAN ale_sigma | pyOCEAN ale_sigma_gls | Clase |
|---|---|---|---|---|---|
| s_in [g/kg] | 36.63 | 38.84 | 31.14 | 38.83 | c |
| s_out [g/kg] | 37.34 | 41.05 | 31.73 | 41.02 | c |
| [(s²)_in]^½ [g/kg] | 36.63 | 38.85 | 31.14 | 38.83 | c |
| [(s²)_out]^½ [g/kg] | 37.38 | 41.06 | 31.73 | 41.03 | c |
| dV [m³/s] | 0.36 | -0.01486 | 0.004265 | 0.01437 | d? |
| dS [g/kg m³/s] | 5.55 | -17.71 | -7.069 | -16.3 | c |
| dS2 [(g/kg)² m³/s] | 237 | -837.2 | -345.8 | -753.8 | c |
| F_surf^{s²} [(g/kg)² m³/s] | 3982 | 5058 | 3912 | 5072 | c |
| Me | 3646 | 4501 | 3809 | 4498 | c |
| Mc | 3678 | 4505 | 3810 | 4502 | c |
| Mp | 3982 | 5058 | 3912 | 5072 | c |
| Mcp | 3982 | 5058 | 3912 | 5072 | c |
| M_model^tot | 3646 | 4501 | 3809 | 4498 | c |
| M_model^phy | 3438 | 3893 | 3206 | 3600 | c |
| M_model^num | 208 | 607.3 | 603.5 | 897.5 | c |
| \|Me − M_tot\|/M_tot | precisión de máquina | 1.2e-11 | 9.7e-11 | 1.6e-10 | — |

**Experimento C** (Q_r, Q_evap, Q_precip, M2 = (0.0, -5.0, 5.0, 0.5))

| Cantidad | GETM | pyOCEAN sigma_gls | pyOCEAN ale_sigma | pyOCEAN ale_sigma_gls | Clase |
|---|---|---|---|---|---|
| s_in [g/kg] | 36.57 | 38.81 | 31.14 | 38.78 | c |
| s_out [g/kg] | 37.12 | 41 | 31.7 | 40.96 | c |
| [(s²)_in]^½ [g/kg] | 36.58 | 38.81 | 31.14 | 38.79 | c |
| [(s²)_out]^½ [g/kg] | 37.18 | 41.01 | 31.7 | 40.97 | c |
| dV [m³/s] | 0.34 | -0.001281 | 0.004149 | 0.02586 | d? |
| dS [g/kg m³/s] | 6.94 | -16.95 | -6.896 | -15.52 | c |
| dS2 [(g/kg)² m³/s] | 339.3 | -788.2 | -341 | -709.7 | c |
| F_surf^{s²} [(g/kg)² m³/s] | 3786 | 4502 | 2300 | 4470 | c |
| Me | 3464 | 3936 | 2203 | 3897 | c |
| Mc | 3495 | 3940 | 2204 | 3902 | c |
| Mp | 3786 | 4502 | 2300 | 4470 | c |
| Mcp | 3786 | 4502 | 2300 | 4470 | c |
| M_model^tot | 3464 | 3936 | 2203 | 3897 | c |
| M_model^phy | 2911 | 3545 | 2203 | 3293 | c |
| M_model^num | 553 | 390.7 | 0.2207 | 604.4 | c |
| \|Me − M_tot\|/M_tot | precisión de máquina | 1.4e-11 | 1.7e-10 | 1.9e-10 | — |

**Experimento D** (Q_r, Q_evap, Q_precip, M2 = (5.0, -5.0, 5.0, 0.5))

| Cantidad | GETM | pyOCEAN sigma_gls | pyOCEAN ale_sigma | pyOCEAN ale_sigma_gls | Clase |
|---|---|---|---|---|---|
| s_in [g/kg] | 24.04 | 21.01 | 26.64 | 20.94 | c |
| s_out [g/kg] | 13.98 | 17.5 | 23.93 | 17.6 | c |
| [(s²)_in]^½ [g/kg] | 24.05 | 21.02 | 26.65 | 20.95 | c |
| [(s²)_out]^½ [g/kg] | 15.24 | 17.68 | 23.94 | 17.78 | c |
| dV [m³/s] | 0.3 | 0.005945 | 0.0001326 | 0.007774 | d? |
| dS [g/kg m³/s] | 8.48 | -79.36 | -0.2275 | -80.12 | c |
| dS2 [(g/kg)² m³/s] | -35.16 | -1434 | -6.908 | -1443 | c |
| F_surf^{s²} [(g/kg)² m³/s] | 946.8 | 631.5 | 1502 | 637.8 | c |
| Me | 2445 | 800.7 | 4685 | 786.6 | c |
| Mc | 2884 | 845.7 | 4686 | 832.7 | c |
| Mp | 2193 | 2288 | 4689 | 2286 | c |
| Mcp | 2628 | 2470 | 4690 | 2480 | c |
| M_model^tot | 2445 | 800.7 | 4685 | 786.6 | c |
| M_model^phy | 2413 | 800.6 | 4606 | 786.5 | c |
| M_model^num | 32 | 0.1059 | 78.79 | 0.153 | c |
| \|Me − M_tot\|/M_tot | precisión de máquina | 1.1e-11 | 4.5e-11 | 1.1e-10 | — |

### Cierre del balance

La identidad exacta de Lorenz (ec. 32), M_model^tot = Me, se cumple en las 12 corridas (última fila de cada tabla) salvo A `ale_sigma_gls`, que no alcanza 5e-10 (ver abajo): \|Me − M\|/M = 1.1e-11 – 7.8e-11 en σ + GLS, 4.5e-11 – 4.3e-10 en ALE-σ y 1.1e-10 – 1.5e-9 en ALE-σ + GLS. Es el piso de redondeo de sumar Σ V S² durante 4.5e6 s; crece como ε Σ V S²/(M T), por eso A, con la M más chica, tiene el residuo más grande en las dos configuraciones. GETM reporta lo mismo ("machine accuracy"). Esto valida juntos la salinidad, el agua dulce real, el balance de varianza discreto (Task 6) y el TEF dentro del modelo (Task 7).

**ALE-σ + GLS (N3 Task 10b, criterio 4 del spec N3: ≤ 5e-10).** B, C y D cumplen (1.6e-10, 1.9e-10, 1.1e-10). A da **1.5e-9**, 3 veces el límite. No es una fuga del GLS: es un sesgo de redondeo del remapeo ALE que ya tenía `ale_sigma`, y A lo amplifica porque su M es la más chica.

| Corrida | M_tot | \|Me − M\| absoluto [(g/kg)² m³/s] | relativo | sesgo por paso del balance de s², en ε Σ V S² | sesgo / Δt [(g/kg)² m³/s] |
|---|---|---|---|---|---|
| A `sigma_gls` | 270.2 | 2.1e-8 | 7.8e-11 | −0.31 (desv. est. 3.4: sin sesgo) | −9.2e-8 |
| A `ale_sigma` | 631.7 | 2.7e-7 | 4.3e-10 | +1.44 (desv. est. 2.4) | 3.1e-7 |
| A `ale_sigma_gls` | 281.0 | **4.3e-7** | 1.5e-9 | +1.33 (desv. est. 2.4) | **4.2e-7** |
| B `ale_sigma_gls` | 4498 | 7.2e-7 | 1.6e-10 | +1.41 | 4.3e-7 |
| C `ale_sigma_gls` | 3897 | 7.3e-7 | 1.9e-10 | — | — |
| D `ale_sigma_gls` | 786.6 | 8.5e-8 | 1.1e-10 | +1.34 | 4.1e-8 |

La columna del sesgo viene de 300 pasos desde el checkpoint final de cada corrida, con la cuenta paso a paso de `tests/test_lorenz_estuary.py::_short_run`: ΔΣ_región V S² contra Δt (fuentes − mezcla) + el flujo de s² del TEF (script de diagnóstico no versionado). La evidencia:
1. **En ALE el balance de s² gana +1.3–1.4 ε Σ V S² en cada paso, siempre con el mismo signo.** En σ el residuo es ruido sin sesgo. El sesgo dividido entre Δt es el error absoluto medio de la ventana solo en A (4.2e-7 contra 4.3e-7); en B (4.3e-7 contra 7.2e-7) y D (4.1e-8 contra 8.5e-8) explica solo 48–60 % del error absoluto, así que la igualdad de A no se generaliza (el resto no está medido).
2. **Es igual con el cierre constante y con el GLS** (1.44 contra 1.33 ε por paso). El error absoluto de A `ale_sigma_gls` es mayor que el de A `ale_sigma` solo porque Δt es menor (59.9 contra 89.8 s): el mismo sesgo por paso, más pasos por segundo.
3. **La sal (primer momento) también gana +1.5 ε por paso; el volumen no** (0.06 ε). Instrumentando `regrid_remap`: el grosor total se conserva (−0.03 ε por paso) y **cada campo remapeado gana +1.4 ε** de contenido (T, S, S², T²).
4. **Mecanismo.** `remap_column` (`core/vertical/remap.py:369-372`; MOM6 hace lo mismo: la sub-celda sin fuente de `intersect_src_tgt_grids`, `MOM_remapping.F90:781-795`, queda asignada a la última celda fuente, :708, y el análogo directo del `remap.py:369-372` es `u_sub(n0+n1+1) = ppoly0_E(n0,2)` con el comentario "only needed when total target column is wider than the source column", `MOM_remapping.F90:939-940`, junto con `xb = min(1., xb)`, :913) trata asimétricamente la diferencia de redondeo entre Σ h_fuente y Σ h_destino. Si el destino es más delgado, la última celda absorbe la fuente sobrante y el contenido se conserva exactamente. Si es más grueso, extiende la última celda con el valor de su borde y **agrega** contenido. La diferencia es de ±1–2 ε D con media cero, pero el contenido solo crece. Con columnas sintéticas (60 capas, Σ h_destino = Σ h_fuente (1 + k ε), k ∈ {−2 … 2}): +1.46 ε de contenido cuando el destino es más grueso y +0.03 ε cuando es más delgado, igual con PCM y PPM_H4.

Clase **(c)**, elección numérica heredada de MOM6, al nivel del redondeo: 1.4 ε por paso son ≈ 2e-11 del contenido de la región en los 74 600 pasos de la ventana. En N3 no se corrigió: hacer exacto el contenido (que la última celda no agregue contenido cuando el destino es más grueso) cambia los bits de toda corrida ALE: rompe la matriz dorada de N1 (criterio 6 del spec N3) y la igualdad con MOM6 de `tests/test_vertical_vs_mom6.py` a nivel de ε. El límite de 5e-10 relativo no escala con este piso: el error absoluto es ≈ 1.4 ε Σ V S²/Δt, independiente de M, así que un caso con M chica lo rebasa aunque el balance esté igual de cerrado.

**Remapeo exacto en contenido (2026-10-30, decisión de Esteban, punto 8 de la lista).** `remap_column` ya no da contenido al grosor extra cuando el destino es más grueso por redondeo (Σ h u exacto en las dos direcciones; MOM6 sigue extendiendo la última celda). Las tablas de arriba son el registro de N3, con el remapeo anterior. A `ale_sigma_gls` repetida con el remapeo exacto (600 ciclos, misma configuración; `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/lorenz_estuary/remap_exact/`):

| A `ale_sigma_gls` | Me | M_tot | \|Me − M\|/M | M_phy | M_num | Fs2_surf | dS2 |
|---|---|---|---|---|---|---|---|
| remapeo de N3 | 281.0 | 281.0 | 1.5e-9 | 263.6 | 17.4 | 5388 | −752 |
| remapeo exacto | 306.6 | 306.6 | **2.9e-11** | 282.5 | 24.1 | 5381 | −1017 |

El balance cierra a 2.9e-11, por debajo de A `sigma_gls` (7.8e-11): criterio 4 cumplido. Me cambia +9 %, pero no por el remapeo: las dos corridas son idénticas (S_reg, S_max, \|u\|max a 3 decimales) hasta el ciclo 10, se separan hacia el ciclo 50 y luego fluctúan alrededor de la misma media (S_reg 32.8–33.8 en ambas, sin desfase sistemático); el forzamiento Fs2_surf coincide a 0.13 % y la diferencia está en el almacenamiento dS2 de la ventana. Es la variabilidad interna de A (episodios convectivos; la corrida de Δt 120 s dio Me 295): una perturbación de nivel ε cambia la trayectoria y la media de 100 ciclos se mueve ~9 %. Por eso el "+4 % sobre σ + GLS" de A está dentro del ruido de la ventana.

### Estacionariedad (series.npz, medias por ciclo de los últimos 100 ciclos)

Las medias por ciclo quitan la marea, que un muestreo cada ¼ de ciclo aliasaría. Veredicto: "deriva" si \|tendencia\|/media ≥ 0.5 % por 100 ciclos; "fluctúa" si la desviación estándar ≥ 0.5 % de la media.

| Corrida | S_region [g/kg] | tendencia [g/kg por 100 ciclos] | tendencia / media | desv. est. | S_max | tendencia de S_max | veredicto |
|---|---|---|---|---|---|---|---|
| A sigma_gls | 33.344 | -0.083 | -0.25 % | 0.341 | 38.84 | -0.487 | fluctúa |
| A ale_sigma | 33.912 | -0.000 | -0.00 % | 0.000 | 37.93 | -0.000 | estacionaria |
| A ale_sigma_gls | 33.212 | -0.069 | -0.21 % | 0.344 | 38.57 | +0.175 | fluctúa |
| B sigma_gls | 24.214 | -0.730 | -3.01 % | 0.214 | 41.85 | -0.550 | deriva |
| B ale_sigma | 24.590 | -0.309 | -1.26 % | 0.090 | 31.84 | -0.243 | deriva |
| B ale_sigma_gls | 24.286 | -0.720 | -2.96 % | 0.211 | 41.79 | -0.531 | deriva |
| C sigma_gls | 24.361 | -0.724 | -2.97 % | 0.212 | 41.79 | -0.506 | deriva |
| C ale_sigma | 25.009 | -0.301 | -1.20 % | 0.088 | 31.82 | -0.233 | deriva |
| C ale_sigma_gls | 24.426 | -0.708 | -2.90 % | 0.207 | 41.70 | -0.515 | deriva |
| D sigma_gls | 6.071 | -3.609 | -59.44 % | 1.045 | 30.00 | +0.000 | deriva |
| D ale_sigma | 16.707 | -0.010 | -0.06 % | 0.015 | 30.00 | -0.000 | estacionaria |
| D ale_sigma_gls | 6.112 | -3.635 | -59.46 % | 1.052 | 30.00 | -0.000 | deriva |

- **A ALE-σ es estacionaria**: S_reg = 33.912 y S_max = 37.926 desde el ciclo ~150, sin cambio en la 4.ª cifra.
- **A σ + GLS fluctúa sin tendencia** (−0.25 % por 100 ciclos). S_reg oscila ± 0.34 g/kg y S_max entre 37.0 y 40.7. Son episodios convectivos: la evaporación vuelve inestable la columna y el k-ε la mezcla a ráfagas. Una ventana de 100 ciclos guarda almacenamiento (dS = 4.5, dS2 = 316); la corrida a 120 s da Me 295 contra 270 (9 %), del mismo orden.
- **B y C, en las dos configuraciones, siguen perdiendo sal**: −0.31 g/kg por 100 ciclos en ALE-σ (−1.2 %) y −0.73 en σ + GLS (−3.0 %). Con un ajuste exponencial de los ciclos 200–600, el tiempo de relajación es ≈ 255–265 ciclos en ALE-σ (faltan ≈ 0.7 g/kg para el equilibrio) y ≈ 540–570 ciclos en σ + GLS (faltan ≈ 3.8–3.9 g/kg; estimación, no medida). 500 ciclos de arranque desde S = 30 no bastan para B y C. GETM tampoco estaba estacionario (dS = 5.6 y 6.9), pero ganaba sal; su estado inicial no está en el artículo.
- **D ALE-σ es estacionaria** (−0.06 %).
- **ALE-σ + GLS sigue a σ + GLS en todo** (N3): A fluctúa igual (desv. est. 0.344 contra 0.341), B y C derivan igual (−2.96 % y −2.90 % contra −3.01 % y −2.97 %), y D tiene el mismo vaciado tras la misma intrusión (S_reg 6.11 contra 6.07, −59 %). La deriva es del cierre, no de la coordenada.
- **D σ + GLS no es estacionaria.** La región se vuelve casi dulce: S_reg baja 0.03–0.07 g/kg por ciclo a lo largo de la corrida (29.96 → 28.31 en los ciclos 0–25, ≈ −0.05 por ciclo en los ciclos 100–460) hasta ≈ 4, y la cabeza tiene S = 0. En los ciclos 478–494, antes de la ventana, entra una intrusión salina que sube S_reg de 3.98 a 8.27 en ~16 ciclos, con \|u\|max de hasta 1.54 m/s (ciclo 482). La ventana de promedio (ciclos 500–600) muestrea **solo el vaciado** después de la intrusión: S_reg 8.0 → 4.4, ≈ −0.036 g/kg por ciclo. Por eso dS = −79 < 0 domina Me. Al final de la ventana \|u\|max vuelve a subir (1.18 m/s en el ciclo 599). Hay saltos menores de S_reg (> 0.1 g/kg en un ciclo) en los ciclos 93, 186, 279, 372, 465 y 558, una recurrencia de ≈ 93 ciclos; la intrusión grande sigue al de 465. Que sea una oscilación de relajación periódica es una **hipótesis**: solo hay un evento grande en 600 ciclos.

### Clasificación de las diferencias

**Exacto en las dos configuraciones:** M_model^tot = Me (arriba); dV ≈ 0 donde GETM tiene 0 (A).

**A, sin marea:**
- σ + GLS reproduce s_in = 30.0 y s_out con un error de 1.3 %. Me = 270 contra 212 (+27 %). Esto es (c): el k-ε es el de CROCO (Canuto A, Galperin) y no el de GOTM, y la ventana guarda almacenamiento de los episodios convectivos (± 9 % solo por eso, ver arriba).
- ALE-σ mezcla 3 × (Me 632). La sal evaporada cruza el transecto (s_in = 31.2). Es (c): ν = κ = 1e-4 m²/s constante, sin marea, está muy por encima de lo que da un k-ε en una salida estratificada estable (hipótesis). En cambio, ALE-σ cumple mejor las aproximaciones: está quieta, así que Mc ≈ Mp ≈ Me.
- La división física/numérica (GETM 103.5/108.8; σ + GLS 248/22; ALE-σ 631.5/0.16) es (c). Advección WENO5 (+ FCT) contra TVD Superbee, y M_num por celda del balance de varianza de pyOCEAN contra el método de Klingbeil et al. (2014). En ALE-σ, κ constante no deja gradientes agudos que la advección pueda mezclar.

**B y C, tapón salino:**
- El tapón está donde GETM lo pone: en el borde oeste de la zona de evaporación, máximo de S medio en x ≈ 34–35 km. Pero σ + GLS lo hace más salado (S medio máx. 40.6, s_in/s_out 38.8/41.0 contra 36.6/37.3), y ALE-σ casi no lo forma (31.6; 31.1/31.7). Es (c), de cierre. Hipótesis: con ν = 1e-4 constante, la fricción de marea es baja y el intercambio fuerte lava el tapón. Con el k-ε, el intercambio es más débil y la sal evaporada se acumula. Además ninguna de las dos está estacionaria (arriba): σ + GLS va hacia ≈ 20.4 g/kg de S_reg.
- dS y dS2 tienen el signo opuesto al de GETM (c). pyOCEAN todavía pierde la sal del arranque con S = 30; GETM ganaba sal desde un estado que el artículo no da.
- Como Q_r + Q_surf = 0, Me ≈ F_surf^{s²} ∝ (S superficial de la zona E)². En B: ALE-σ +4 %, σ + GLS +23 % (c, sigue a la S superficial). En C, ALE-σ queda −36 % (F −39 %): κ constante mezcla la capa superficial delgada de agua de lluvia que GETM conserva (el artículo, l. 1057-1061, atribuye a esa capa el aumento de M_num de GETM en C). σ + GLS queda +14 % (c).
- **M_num de B y D en ALE-σ está casi todo en la celda del río.** Al repetir un ciclo desde el checkpoint del ciclo 575 (diagnóstico ad hoc, script no versionado), 589 de 596 (B) y 78 de 79 (D) caen en la celda i = 499, donde el río entra con S = 0 a agua de S ≈ 10 (B) o ≈ 3.5 (D). Es ≈ Q_r s_cabeza², la mezcla de dos masas de agua en la celda de entrada; el diagnóstico la cuenta como numérica porque ocurre en el flujo advectivo de la cara del río. En σ + GLS B, 117 de 585 están en esa celda y el resto en los frentes del interior (i = 200–500). En ALE-σ, la sal llega a la cabeza del río porque el intercambio es fuerte; en GETM, por lo visto, no (M_num 208). Es (c). En C, sin río, ALE-σ tiene M_num = 0.2.
- **dV (d?, abierta).** GETM da 0.30–0.36 m³/s en B–D: +10–12 cm de nivel medio en la región por 100 ciclos, con η prescrita en la frontera. pyOCEAN da ≤ 0.026 m³/s (0.026 es C `ale_sigma_gls`; el resto ≤ 0.015). No hay una explicación física evidente para una subida así en un estado cuasi periódico. Hipótesis: es una propiedad del promedio o de la corrida de GETM. No cambia nada en pyOCEAN.

**D, estuario clásico:**
- ALE-σ: s_in − s_out = 2.7 contra 10.1, y Q_in = 44 contra ≈ 7 m³/s (Knudsen con los valores de GETM). La circulación gravitacional es ≈ 6 × más fuerte y la región más salada; Me = 4685 (+92 %). Es (c). Hipótesis: es la escala clásica u_e ∝ g′H³/(ν_v L), con ν constante y bajo comparado con el ν de marea de un k-ε.
- σ + GLS: Me = 801 (−67 %) porque la ventana muestrea solo el vaciado tras la intrusión de los ciclos 478–494 (dS = −79). Su Mp = 2288, que no ve el almacenamiento, queda cerca de GETM (2193); puede ser coincidencia. Es (c). Hipótesis: el k-ε de CROCO deja menos intrusión salina que el de GOTM con este forzamiento, y el sistema queda cerca de un umbral (intrusiones episódicas).

Ninguna diferencia resultó (a). La (b) que asoma es que ALE no tiene cierre turbulento (`core/model/preflight.py:130`: ALE solo admite `closure` 'constant' o 'none'): la mayoría de las diferencias de ALE-σ vienen de ν = κ constantes, una elección (c) de este caso que la física que falta (b) obliga a tomar.

**ALE-σ + GLS (N3 Task 10b, criterio 4 del spec N3).** La (b) de arriba ya está: el GLS estratificado corre en ALE (N3 Task 5) con la ley de pared de σ (Task 4, `hbbl` = 0.05 m). Resultado: **ALE-σ + GLS ≈ σ + GLS en A–D**, así que el cierre, no la coordenada, causaba las diferencias de ALE-σ.

| | A | B | C | D |
|---|---|---|---|---|
| Me: ALE-σ + GLS / σ + GLS / GETM | 281 / 270 / 212 | 4498 / 4501 / 3646 | 3897 / 3936 / 3464 | 787 / 801 / 2445 |
| ALE-σ + GLS contra σ + GLS | +4.0 % | −0.07 % | −1.0 % | −1.8 % |
| s_in / s_out (ALE-σ + GLS; σ + GLS) | 30.01 / 33.33; 30.01 / 33.10 | 38.83 / 41.02; 38.84 / 41.05 | 38.78 / 40.96; 38.81 / 41.00 | 20.94 / 17.60; 21.01 / 17.50 |
| Q_in [m³/s] (ALE-σ + GLS; σ + GLS; ALE-σ) | 53.4; 52.2; 116.5 | 7.69; 7.76; 12.2 | 7.62; 7.71; 12.5 | 2.31; 2.28; 44.1 |
| S_max, media de la ventana (ALE-σ + GLS; σ + GLS) | 38.57; 38.84 | 41.79; 41.85 | 41.70; 41.79 | 30.00; 30.00 |
| M_num (ALE-σ + GLS; σ + GLS; GETM) | 17; 22; 109 | 898; 607; 208 | 604; 391; 553 | 0.15; 0.11; 32 |

- **S máximo, Me, Q_in y s_in:** ALE-σ + GLS queda a 0–4 % de σ + GLS (criterio 4: "se acerca a GETM y a nuestro σ + GLS"). Contra GETM se acerca mucho respecto a ALE-σ constante (A: Me +32 % contra +198 %; s_in 30.01 contra 31.19 = GETM 30.00; B y C: el tapón salino aparece; D: Q_in 2.3 contra 44 m³/s), pero hereda las diferencias de σ + GLS con GETM: tapón 3–4 g/kg más salado en B y C, y el vaciado de D tras la intrusión (Me −68 %). **(c)**, del k-ε de CROCO contra el de GOTM y de la ventana (abajo, punto 5 de la lista); la misma clase que en σ + GLS.
- **A: +4 % sobre σ + GLS.** Dentro de la variabilidad de A (episodios convectivos, ± 9 % entre ventanas; dS de signo opuesto, −10.9 contra +4.5). **(c).**
- **División física/numérica en B y C.** El total es el mismo, pero ALE-σ + GLS pone más en M_num (898 contra 607 en B, 604 contra 391 en C) y menos en M_phy. El remapeo aporta ≈ 15 (checkpoint de la ventana); el resto es la advección. Hipótesis: el FCT de ALE (Zalesak, cotas en x, y, k ± 1) recorta los frentes del tapón y de la capa de lluvia, mientras que σ usa WENO5 sin FCT. **(c)**, no separado (punto 9 de la lista).
- **Cierre del balance:** ver "Cierre del balance" arriba (A 1.5e-9 con el remapeo de N3, sesgo de redondeo del remapeo, (c); 2.9e-11 con el remapeo exacto del 2026-10-30).

### El escalón de u medio en B ALE-σ

En la sección de u medio de B ALE-σ (`section.npz`, `u_mean`) hay escalones verticales nítidos, y S no muestra ninguno. Medidos celda por celda, no están en el transecto (i = 167, x = 33.4 km) sino a los dos lados. Entre i = 145 y 150 (x ≈ 29–30 km), u de la capa superior baja de 3.4 a 1.7 cm/s y la del fondo sube de −2.6 a −1.8 cm/s. Entre i = 186 y 194 (x ≈ 37–39 km) hay otro escalón. En σ + GLS el mismo cambio se reparte en ~15 celdas.

**Conclusión: es físico, no un artefacto del promedio, y lo afila la configuración de ALE-σ.** La evidencia:
1. **No es el sumidero de volumen de la evaporación.** La u media integrada en la vertical no tiene escalón: cambia linealmente desde i = 167, como pide un sumidero uniforme de 0.03 m³/s por celda. El escalón es baroclínico (la superficie y el fondo saltan con signos opuestos).
2. **No viene de promediar en capas σ móviles.** Al repetir un ciclo desde el checkpoint del ciclo 575 (diagnóstico ad hoc, script no versionado; también para los puntos 3 y 4), el campo instantáneo tiene un frente real en u. En la estoa de fin de vaciante la capa superior pasa de 42 a 22 cm/s en 3 celdas (i = 146–149), con S superficial invertida (más salada arriba que abajo, ΔS = +0.03 a +0.13 g/kg). Promediar en z fijo no quita un frente que existe en el campo instantáneo.
3. **Es el frente de evaporación llevado por la marea.** La posición del frente de \|∂u/∂x\| máximo recorre i = 146 → 190 → 146 en el ciclo. Son 8.8 km, la excursión de marea de la capa superior (2U/ω con U ≈ 0.6 m/s ≈ 8.5 km), centrada en i ≈ 168, el borde de la zona de evaporación. El frente pasa la mayor parte del ciclo en los extremos (histograma de posición: 127 de 498 pasos en i = 144–151 y 86 en i = 188–191), donde se detiene para invertir. Por eso el promedio euleriano tiene sus escalones justo ahí. S media no tiene escalón porque el contraste de S en el frente es de solo 0.05–0.1 g/kg, mientras que la cizalla salta 10–20 cm/s.
4. **Lo afila el cierre constante.** Con κ = 1e-4 constante y sin ajuste convectivo, la columna estáticamente inestable del frente sobrevive. Con A_h = 0, la velocidad tiene oscilaciones de escala de malla junto al frente (42.5 → 36 → 26 → 22 cm/s). En σ + GLS, el k-ε mezcla la inestabilidad y el frente medio queda suave.

Clase: (c) en este caso. Para pyOCEAN, falta un ajuste convectivo o un cierre en ALE (b) (lista de abajo).

### Antes y después de Task 8c (GLS sin y con flotabilidad)

| | antes de 8c (`*_sigma_gls_pre8c`, Δt 120 s) | después (`*_sigma_gls`, Δt 60 s) |
|---|---|---|
| A | S_reg 30 → 83 lineal, S_max 191.7, \|u\|max 0.04 m/s: la región no exporta la sal evaporada (hay intercambio, Q_in = 16.5 m³/s y s_in = 35.2, pero débil; S_reg crece lineal). Me = M = 17 703 (identidad 8e-12), dominada por el almacenamiento (dS = 140) | S_reg 33.3 ± 0.3, S_max 38.8; s_in = 30.0 = GETM; Me 270 |
| B | circulación residual de dos celdas; Q_in ≈ 0 en el transecto, s_in = −1785 y Me = NaN (TEF degenerado), S_max 55 | tapón salino en x ≈ 35 km, circulación inversa en el transecto como describe el artículo (l. 1041); Me 4501 |
| C | detenida en el ciclo 370 con S_max 56.6 | Me 3936 |

Sin flotabilidad, el k-ε ve N² = 0 y da ν ≈ 1e-3–1e-2 m²/s aun con Ri ≈ 80. Eso frena la circulación de intercambio que debe exportar la sal evaporada, y la sal crece sin límite. Con la producción de flotabilidad B = −κ_t N² y κ_t sobre S (Task 8c), la mezcla cae a ≈ 1e-5 bajo la haloclina y convecciona donde la evaporación vuelve inestable la columna. Entonces A llega a un estado acotado con s_in como el de GETM. Era física que faltaba (b) y Task 8c la agregó. Su costo fue el límite de Δt de arriba: menos ν permite chorros más fuertes.

### Qué debe mejorar pyOCEAN

1. ~~**Cierre turbulento en ALE (b).**~~ **Hecho en N3** (GLS estratificado en ALE, Task 5; KPP, Task 7). Las cuatro corridas `ale_sigma_gls` quedan a 0–4 % de σ + GLS (arriba): las diferencias de ALE-σ con GETM eran del cierre constante. Lo que queda contra GETM es lo de σ + GLS (puntos 5 y 6).
2. **Ajuste convectivo para el cierre constante (b).** Sin él, una columna estáticamente inestable sobrevive (el frente de evaporación de B, arriba). Una opción mínima: κ grande donde N² < 0, como `convect` de MITgcm o el `Kv_conv` de MOM6 [verificar].
3. **Límite de estabilidad del momento horizontal con A_h = 0 (c, robustez).** (i) Smagorinsky ya existe (spec 2026-10-30): con C = 0.15, D a T_M2/498 termina 150 ciclos, pero con ráfagas de hasta \|u\| Δt/Δx = 0.92 (sección de la búsqueda de Δt). Faltan (ii) y (iii). σ + GLS revienta a \|u\| Δt/Δx ≈ 0.75–0.9 (AB3 + WENO5 en forma de flujo, modo de 3–4 Δx, ×1.3 por paso), y el aviso llega solo como la matriz singular del solver y el BLOW-UP del runner, sin diagnóstico del Courant ni del modo de malla. Hacen falta: (i) una opción de viscosidad biarmónica o Smagorinsky pequeña (MOM6, ROMS), (ii) un respaldo tipo `truncate_cfl` o al menos un aviso de Courant advectivo en σ, como el de ALE, y (iii) que el post-mortem atrape el modo de malla antes del fallo del solver.
4. **Separar la mezcla de entrada del río en el balance de varianza (c, diagnóstico).** En ALE-σ, ≈ 98 % de M_num de B y D es la mezcla del agua del río (S = 0) con la celda de entrada (≈ Q_r s_cabeza²). Conviene un término propio (p. ej. `S_bdy_in`) para que M_num mida solo el error de los esquemas de advección y remapeo, y comparar así con el M_num de GETM.
5. **Verificar el k-ε estratificado contra GOTM (c, hipótesis).** σ + GLS hace el tapón 3–4 g/kg más salado que GETM y casi no deja entrar sal en D. Un caso de columna con marea y estratificación, comparado con GOTM (Canuto A, mismos parámetros), separaría el efecto del cierre del de la advección (WENO5 contra Superbee).
6. **Arranque de los benchmarks lentos (c, configuración).** B y C no llegan a estado cuasi periódico en 500 ciclos (relajación de ≈ 255–570 ciclos). `compare.py` ahora reporta la estacionariedad. Para una comparación final conviene seguir B y C desde su checkpoint hasta que la tendencia sea < 0.5 % por 100 ciclos, o arrancar de un estado más cercano. El artículo fija 500 ciclos, pero no el estado inicial. D σ + GLS necesita además una ventana que cubra varias intrusiones (recurrencia de ≈ 93 ciclos en los eventos menores, uno solo grande en 600 ciclos), es decir una corrida mucho más larga, antes de que su columna de la Tabla 2 signifique algo.
7. **dV de GETM (d?, abierta).** Antes de usar las filas dV/dS de GETM como referencia, preguntar o verificar cómo se promedia dV en GETM (+10–12 cm por 100 ciclos en B–D).
8. ~~**Remapeo exacto en contenido (c, N3 Task 10b).**~~ ✅ 2026-10-30: `remap_column` ya no agrega contenido cuando Σ h_destino supera a Σ h_fuente por redondeo (antes +1.4 ε Σ V S² por paso, como MOM6). A `ale_sigma_gls` cierra a 2.9e-11 (era 1.5e-9). Cambiaron los bits de toda corrida ALE a nivel 1e-14 (matriz dorada de N1 regenerada, 18 casos; σ idéntico); la paridad con MOM6 de `tests/test_vertical_vs_mom6.py` sigue dentro de su tolerancia de 1e-11. Ver "Cierre del balance".
9. **FCT y mezcla numérica (c, hipótesis).** Con el mismo cierre y la misma M total, ALE-σ + GLS tiene 1.5 × la M_num de σ + GLS en B y C. Una corrida corta de B con `tracer_limiter` apagado en ALE (o FCT en σ) separaría el efecto del limitador del de la coordenada.
10. **Costo del GLS en ALE.** 0.020–0.022 s/paso contra 0.0134 de σ + GLS y 0.016 de ALE-σ (ventana de producción). El Δt lo fija el chorro superficial de σ + GLS (T_M2/746, la búsqueda de 10a); el resto es el GLS en ALE (fusión de capas delgadas, remapeo de k y ε).

Pendiente fuera de N2a: figura de las secciones medias de S y u, como la Fig. 2 del artículo; los datos están en `section.npz`.

