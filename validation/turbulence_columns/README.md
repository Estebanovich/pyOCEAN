# Columnas de turbulencia: GLS y KPP de pyOCEAN contra CROCO (`SINGLE_COLUMN`)

N3, tarea 8 (spec `docs/superpowers/specs/2026-10-29-n3-turbulence-closures-design.md`, criterios 1 y 3, ítem 6.1).
Los siete casos de una columna de CROCO, con sus valores exactos (`docs/design/n2c_refs/croco_column_cases.md`,
`cases.py`). Nada se ajustó.

## Cómo correr

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
# CROCO (≈ 15 s de compilación y 1-20 s de corrida por caso)
validation/turbulence_columns/croco/build_croco.sh kato_phillips gls     # gls | kpp | kppsfc | kppraw
validation/turbulence_columns/croco/run_croco.sh   kato_phillips gls
# pyOCEAN (una corrida a la vez)
python -m validation.turbulence_columns.run_pyocean kato_phillips sigma gls [--production energy]
python -m validation.turbulence_columns.run_pyocean --matrix             # las 26 corridas, ≈ 50 min
python -m validation.turbulence_columns.compare                          # todas las tablas de abajo
```

Salidas: `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/turbulence_columns/{croco,pyocean}/` (nunca en el repo). Ejecutables de
CROCO: `~/pyOCEAN_refs/croco_runs/scm/<caso>_<cierre>/`.

## Configuración

- **CROCO** (`croco/build_croco.sh`): el `cppdefs.h` del código con `#undef REGIONAL`, `#define SINGLE_COLUMN` y una
  sola clave de caso; `param.h`, `ana_*.F`, `analytical.F` y `prsgrd.F` sin tocar; el `jobcomp` de `tidal_flat3d`
  (cambios para arm64 de `validation/thacker/README.md`). El `croco.in.<CASO>` de `TEST_CASES` se pasa en la línea de
  comandos (evita la errata `KATO_PHILLIPS` de `read_inp.F:127`); solo cambian las rutas de salida, `NINFO` y los
  campos del historial (Akv, Akt, HBL, esfuerzos, `TKE GLS Lscale`). `SINGLE_COLUMN` no define `AVERAGES` (solo el
  bloque REGIONAL, `cppdefs.h:310`), así que la referencia son las fotos horarias de `scm_his.nc`.
  - KPP: `kpp` = `#undef GLS_MIXING`, `LMD_MIXING` con el preset regional (`LMD_SKPP`, `LMD_BKPP`, `LMD_RIMIX`,
    `LMD_CONVEC`, `LMD_NONLOCAL`, `cppdefs.h:255-259`); `kppsfc` = solo `LMD_SKPP` + `LMD_NONLOCAL` (el análogo del
    `'kpp'` de pyOCEAN); en el ciclo diurno además `DIURNAL_SRFLUX`, que solo pone Ric = 0.15
    (`lmd_skpp2005.F:97-103`, McWilliams et al. 2009).
  - **Error de CROCO corregido localmente en `kpp`/`kppsfc` (clase d).** Con EOS lineal, `alfabeta.F:133` (y `:138`)
    devuelve α = |Tcoef| en kg m⁻³ K⁻¹; la rama no lineal divide entre ρ₀ (`alfabeta.F:116-118`) y
    `lmd_skpp2005.F:180-186` necesita α en K⁻¹ (B₀ = g α stflx en m² s⁻³). El flujo de flotabilidad de KPP sale
    ρ₀ = 1024 veces mayor: en Willis–Deardorff la convección mezcla los 50 m en 6 h (h_bl = 50 m, SST 19.37 °C, contra
    15.35 m y 20.77 °C corregido; `kppraw` guarda la versión sin corregir). Solo KPP llama `alfabeta_tile`
    (`lmd_skpp1994.F:177`, `lmd_skpp2005.F:167`); la versión 1994 tiene el mismo problema (probado). En Kato–Phillips
    no hay flujo de calor y no cambia nada. El directorio de corrida lleva una copia de `alfabeta.F` con α = |Tcoef|/ρ₀.
- **pyOCEAN** (`cases.py`): caja doblemente periódica 5 × 5 (Δx = 2 km), σ uniforme, Δt = 30 s, EOS lineal
  (ρ₀ = 1024, en EKBBL 1.22; Tcoef = 0.2, T₀ = 16), c_p = 3985, GLS k-ε Canuto A estratificada con los pisos de CROCO
  (ν ≥ 1e-5, κ_t ≥ 1e-6), KPP con los valores de CVMix (`KPPParams()`, sin fondo). `z0` = Zob (ALE: `hbbl` = ½ Δz, así
  la capa del fondo lleva sola el esfuerzo, como en σ). Coordenadas: `sigma`, `zstar` (ALE z*, capas fijas) e
  `hybrid` (ALE HYCOM1 con las isopicnas iniciales como blancos y la malla σ como piso z*: en la capa de mezcla las
  capas quedan en z*, en su base las interfaces cuya densidad se mezcló colapsan — malla que se mueve y capas casi
  vacías).
  - Dos forzamientos de CROCO no tienen argumento en `step()` y los añade el arnés sin tocar el modelo: la fuerza de
    cuerpo de los FORCED_* (`prsgrd.F:455-462`: ∂v/∂t += f U_G, −g∂ζ/∂y o −g∂ζ/∂y cos ωt) como término explícito con
    nombre `"body_force"` (último de `Momentum.terms`, pasa por AB3 como el gradiente de presión de CROCO), y el
    pequeño flujo de calor del fondo de los tres casos estratificados (`analytical.F:133-136`), que entra en la capa
    más baja con más de 1 mm (`H_MASSLESS`, el piso de capas delgadas de la GLS; en σ y z* la del fondo), para que una
    capa de fondo casi vacía del híbrido no reciba Δt·btflx/h₀ → ∞. `build` verifica que el gancho de la fuerza de
    cuerpo funcionó (el último término es `"body_force"` y cambia la tendencia de v en F) y si no, falla.
  - T inicial de los FORCED_* = 20 °C (rama por defecto, `ana_initial.F:1222`; `croco_column_cases.md` §4 decía T₀ = 16,
    sin efecto: T es uniforme).

**Métricas** (`compare.py`): D_m = profundidad de la primera interfaz, desde arriba, con k < 1e-5 m² s⁻²
(Warner et al. 2005), con log k interpolado; D_m(T) = interfaz con el mayor ∂T/∂z (para KPP, sin k), sin contar las
capas de menos de 1 mm ni el 10 % inferior de la columna (allí el flujo de calor del fondo de CROCO enfría la celda del
fondo y da un gradiente mayor que un salto de arrastre débil: sin esa exclusión CROCO-GLS daba 49.5 m en Kato y 149 m en
el ciclo diurno); ley de Price
D_m = 1.05 u* N₀^(−1/2) t^(1/2) (Warner et al. 2005, ec. 60; 1.05 = (2 R_v)^{1/4} con el número de Richardson global
R_v ≈ 0.6 de Price 1979, l. 30, 481 y 599 de la copia local); errores de perfil = RMS(pyOCEAN − CROCO)/rango de CROCO.

## Resultados

### 1. Kato–Phillips (criterio 1: D_m a ±5 % de CROCO y de Price)

| corrida | D_m 6 h | 12 h | 24 h | 30 h | vs Price (30 h) | vs CROCO (30 h) |
|---|---|---|---|---|---|---|
| Price 1979 | 15.43 | 21.82 | 30.86 | 34.51 | — | — |
| CROCO GLS | 15.88 | 22.61 | 32.11 | 36.04 | +4.4 % | — |
| pyOCEAN σ GLS | 15.88 | 22.61 | 32.11 | 36.03 | +4.4 % | −0.0 % |
| pyOCEAN z* GLS | 15.88 | 22.61 | 32.11 | 36.03 | +4.4 % | −0.0 % |
| pyOCEAN híbrido GLS | 15.91 | 22.70 | 32.16 | 36.07 | +4.5 % | +0.1 % |
| pyOCEAN σ / z* GLS, P = `energy` | 15.88 | 22.61 | 32.11 | 36.03 | +4.4 % | −0.0 % |
| pyOCEAN híbrido GLS, P = `energy` | 15.91 | 22.71 | 32.26 | 36.07 | +4.5 % | +0.1 % |

**Pasa** en las tres coordenadas: a 0.1 % de CROCO y +4.4 % de Price (CROCO mismo da +4.4 %). Perfiles de T, u, k, ε,
ν, κ_t a 30 h: σ y z* = CROCO a < 1e-3 del rango; híbrido ≤ 0.8 %.

### 2. Los otros seis casos GLS contra CROCO (criterio 1: perfiles de T, u, k, ε)

RMS(pyOCEAN σ − CROCO)/rango(CROCO) a la mitad / al final de la corrida (z* da lo mismo, §3):

| caso | T | u | v | k | ε | ν | κ_t | diagnóstico (CROCO / pyOCEAN σ) |
|---|---|---|---|---|---|---|---|---|
| willis_deardorff | 0.000/0.000 | — | — | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | 0.000/0.000 | D_m(k) 72 h 12.46 / 12.46 m, SST 20.85 / 20.85 |
| diurnal_cycle | 0.000/0.000 | — | — | 0.002/0.002 | 0.001/0.001 | 0.004/0.004 | 0.003/0.003 | SST día 8: rango 0.340 / 0.339 K, media 21.947 / 21.947; D_m 06 h 26.68 / 26.67 m |
| forced_ekbbl | — | 0.000/0.000 | 0.000/0.000 | 0.003/0.005 | 0.000/0.001 | 0.000/0.000 | 0.012/0.022 | ū 16.66 / 16.66 m/s; ángulo en la capa del fondo **75.8° / 76.0°** (antes de la tarea 9b: 62.8°); k máx 0.131 / 0.130 |
| forced_dbleek | — | 0.000/0.000 | 0.000/0.000 | 0.001/0.001 | 0.000/0.000 | 0.001/0.003 | 0.001/0.003 | ū 0.84 / 0.84 mm/s, v̄ 5.07 / 5.07 mm/s, ángulo 63.1° / 63.2° (antes 63.4°) |
| forced_nonrotbbl | — | 0.000/0.000 | 0.009/0.009 | 0.000/0.000 | 0.001/0.001 | 0.001/0.001 | 0.001/0.001 | v̄ −0.1341 / −0.1331 m/s (ley log: −0.126); ν máx 4.13e-3 / 4.13e-3 |
| forced_oscnonrotbbl | — | 0.000/0.000 | 0.007/0.002 | 0.048/0.029 | 0.034/0.009 | 0.010/0.008 | 0.011/0.009 | v̄ último ciclo: amplitud 0.1113 / 0.1109 m/s, fase −125.5° / −125.5° |

Teoría de convección penetrante (Willis–Deardorff, modelo de salto de orden cero, h² = 2(1 + 2A) B₀ t / N²
[verificar]): 11.3 m sin arrastre (A = 0) y 13.3 m con A = 0.2; CROCO y pyOCEAN dan 12.5 m (k) y 13 m (T).

**Pasa** en los seis casos con errores ≤ 1 % del rango en u, v, T, ν (el ciclo diurno ≤ 0.4 %, la capa oscilante ≤ 5 %
en k y ε a media corrida, en las fases de k pequeño; FORCED_EKBBL κ_t 1.2/2.2 %). FORCED_EKBBL no pasaba en la capa
del fondo (62.8° contra 75.8°, clase a): el arrastre log usaba la velocidad de la propia componente. Corregido en la
tarea 9b (rapidez completa √(u² + v̄²), CROCO get_vbc.F:517-528): 76.0° en σ y z*.

### 3. σ contra ALE (criterio 3)

max |ALE − σ| / max |σ| en toda la corrida:

| caso | coord. | T | u | v | k | ε | ν | κ_t |
|---|---|---|---|---|---|---|---|---|
| kato_phillips | z* | 8.5e-13 | 2.9e-12 | 0 | 5.2e-12 | 1.3e-12 | 2.2e-10 | 3.7e-10 |
| willis_deardorff | z* | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| diurnal_cycle | z* | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| forced_ekbbl | z* | 0 | 1.1e-13 | 2.2e-13 | 4.8e-14 | 7.2e-14 | 0 | 3.6e-13 |
| forced_dbleek | z* | 3.4e-15 | 5.2e-12 | 4.0e-12 | 4.2e-12 | 2.2e-13 | 7.0e-11 | 5.9e-11 |
| forced_nonrotbbl | z* | 1.9e-12 | 0 | 1.1e-12 | 3.0e-12 | 2.2e-12 | 1.8e-10 | 4.0e-10 |
| forced_oscnonrotbbl | z* | 8.3e-13 | 0 | 5.6e-12 | 1.1e-11 | 5.1e-12 | 1.5e-09 | 3.0e-09 |
| kato_phillips | híbrido | 1.9e-4 | 1.9e-3 | 0 | 1.6e-2 | 5.7e-3 | 4.5e-2 | 4.0e-2 |
| willis_deardorff | híbrido | 3.0e-3 | 0 | 0 | 1.2e-1 | 1.1e-1 | 1.8e-1 | 2.5e-1 |
| diurnal_cycle | híbrido | 1.5e-3 | 0 | 0 | 1.9e-1 | 9.7e-2 | 3.8e-1 | 4.8e-1 |

- **z* con capas fijas = σ a redondeo** (≤ 3e-9 relativo en κ_t, que hereda el redondeo de k²/ε cerca del piso; el plan
  acepta hasta 1e-8). **Pasa.**
- **Híbrido (capas que se mueven):** k ≥ 1e-10 y ε ≥ 1e-12 en todas las horas (pisos de CROCO), nada no finito. Los
  máximos del 10-50 % en k, ν, κ_t están en la base de la capa de mezcla, donde las interfaces colapsan: es el máximo
  sobre una interpolación a niveles fijos de un perfil con un salto. Las magnitudes integrales: D_m en Kato a 0.1 % de σ,
  Willis −0.36 m (k) / +1 m (T, una celda), ciclo diurno D_m 06 h día 8 26.42 contra 26.67 m. RMS de D_m híbrido − σ
  en el ciclo diurno 0.72 m (8 días) con el umbral de fusión de 1 mm de la tarea 5; la mayor diferencia, −5.1 m a las
  08 h del día 8, cuando la convección nocturna se apaga: había una capa de 1.0 mm (= `h_min` del regrid, no fusionada
  porque la regla es h < 1 mm) justo en la base de la capa de mezcla y ν cayó a 1.9e-4 bajo ella (σ: 9.3e-3).
  **Tarea 9c** (umbral por columna h_thin = min(10 cm, max(1 mm, 1 % de D/N)) = 1 cm aquí, y las capas delgadas de
  superficie como capa fusionada propia, como MOM6; `core/components/closure.py`, `thin_layer_threshold`,
  `active_layers`): RMS 0.48 m; el día 8 a las 08 h D_m = 25.8 m (σ 26.2, antes 21.1); la mayor diferencia es ahora
  −2.85 m a las 08 h del día 4 (17.2 contra 20.0 m), igual que antes de la tarea 9c: dos capas de 1 mm en la base de
  la capa de mezcla, ya fusionadas — es la malla distinta (clase c), no el umbral.
- La conservación de ∫k en el remapeo está probada a 1e-12 en `tests/test_gls_ale.py`; aquí no hay un diagnóstico
  aparte de mezcla numérica de k (anti-objetivo 3): la diferencia híbrido − σ en D_m (≤ 0.1 % en Kato con regrid) es la
  cota disponible.

### 4. KPP (decisión 8: contra la teoría; CROCO-KPP solo como referencia, diferencias de constantes = clase c)

| caso | corrida | h_bl final [m] | D_m(T) final [m] | SST final [°C] |
|---|---|---|---|---|
| kato_phillips (Price 34.51 m a 30 h) | pyOCEAN σ / z* KPP | 33.23 | 32.50 | 15.222 |
| | CROCO-KPP (preset) | 41.36 | 43.50 | 15.014 |
| | CROCO-GLS | — | 36.00 | 15.264 |
| willis_deardorff (teoría 11.3–13.3 m) | pyOCEAN σ / z* KPP | 13.36 | 14.00 | 20.855 |
| | CROCO-KPP (corregido) | 15.35 | 16.00 | 20.772 |
| | CROCO-GLS | — | 13.00 | 20.852 |
| diurnal_cycle (fin = medianoche del día 8) | pyOCEAN σ / z* KPP | 19.85 | 27.00 | 21.857 |
| | CROCO-KPP (corregido, Ric 0.15) | 37.46 | 37.00 | 21.809 |
| | CROCO-GLS | — | 28.00 | 21.851 |

Kato–Phillips h_bl a 6/12/24/30 h: pyOCEAN 16.6 / 22.2 / 30.1 / 33.2 m (Price 15.4 / 21.8 / 30.9 / 34.5): **−3.7 % de
Price a 30 h, pasa ±5 %**. CROCO-KPP 19.7 / 27.6 / 36.9 / 41.4 (+20 %; `kppsfc` igual, 43.9). Ciclo diurno, día 8 a
las 06 h: pyOCEAN KPP h_bl 27.3 m (= GLS 27 m), CROCO-KPP 37.4 m; rango de la SST 0.346 (pyOCEAN KPP), 0.314
(CROCO-KPP), 0.340 K (GLS). σ = z* en KPP (las tablas son idénticas a los decimales mostrados; la prueba de 1e-10 está en
`tests/test_kpp_model.py`).

### 5. `gls_production="energy"` (Burchard 2002) — recomendación para Esteban

En Kato–Phillips (Δz = 0.5 m, Δt = 30 s) `energy` y `croco` dan el mismo D_m a < 1 cm en σ y z* (0.1 m en el
híbrido); los campos difieren ≤ 0.4 % (k), 0.12 % (ν), 2e-5 (T). El caso no distingue las dos formas: con capas finas
y Δt corto la energía que quita la viscosidad implícita y ν S² son casi iguales. **Recomendación:** dejar `croco` por
defecto (conserva el bit a bit de la matriz dorada, criterio 6) y no cambiarlo por este caso; volver a decidir con
Lorenz y DOME (capas gruesas, Δt de 600 s), donde Burchard (2002) espera la diferencia.

### 6. Tiempos de reloj (columna periódica 5 × 5, un proceso, ≤ 5 hilos Numba)

| corrida | pasos | σ GLS | z* GLS | híbrido GLS | σ KPP | z* KPP |
|---|---|---|---|---|---|---|
| kato_phillips (N = 100) | 3600 | 35 s (9.8 ms/paso) | 54 s (15.0) | 54 s (15.0) | 54 s (15.0) | 69 s (19.1) |
| willis_deardorff (50) | 8640 | 30 s (3.4) | 53 s (6.1) | 50 s (5.8) | 51 s (5.9) | 68 s (7.9) |
| diurnal_cycle (150) | 23040 | 250 s (10.9) | 379 s (16.4) | 375 s (16.3) | 377 s (16.4) | 509 s (22.1) |
| forced_ekbbl (40) | 2880 | 18 s (6.4) | 23 s (8.1) | — | — | — |
| forced_dbleek (25) | 2880 | 13 s (4.4) | 16 s (5.4) | — | — | — |
| forced_nonrotbbl (50) | 3600 | 21 s (5.8) | 31 s (8.5) | — | — | — |
| forced_oscnonrotbbl (50) | 7200 | 41 s (5.7) | 60 s (8.4) | — | — | — |

CROCO: 0–19 s por caso (un núcleo). pyOCEAN es 10-25 veces más lento en estas columnas; buena parte es el halo de 13
celdas de la advección periódica (31 × 31 celdas para una columna 5 × 5).

## Clasificación de las diferencias

| diferencia | clase | evidencia |
|---|---|---|
| FORCED_EKBBL: ángulo de la capa del fondo 62.8° contra 75.8°, k máx −7 %, κ_t 4 % | **(a) nuestro** | El arrastre log de pyOCEAN usa la velocidad de la propia componente en cada cara (C_d \|u\| u, C_d \|v\| v), CROCO usa \|U\| = √(u² + v²) (nota del ledger, tarea 4). En la capa del fondo CROCO tiene (u, v) = (0.60, 2.36) m/s, pyOCEAN (1.18, 2.31): con \|u\| = 1.18 en vez de \|U\| ≈ 2.4 el esfuerzo sobre u sale a la mitad y u del fondo al doble. En DBLEEK (u ≈ v pequeños) el efecto es 0.3°, en los casos sin rotación (solo v) cero. **Corregido en la tarea 9b**: σ y z* dan 76.0° (CROCO 75.8°), k máx 0.130 (0.131); DBLEEK 63.2° (63.1°). |
| CROCO-KPP con EOS lineal: B₀ × ρ₀ | **(d) CROCO** | `alfabeta.F:133` vs `:116-118`; corregido localmente (arriba). |
| KPP de pyOCEAN contra CROCO-KPP: h_bl −20 % (Kato), −13 % (Willis), −27 % (ciclo diurno 06 h) | **(c)** | CVMix (Ri_c = 0.3, sin núcleo Kern) contra el KPP 2005 "if-less" de CROCO (Ric = 0.45, 0.15 diurno, `lmd_skpp2005.F:95-117`) y el preset con Ri interior y convección. pyOCEAN queda más cerca de Price (−3.7 %) y de la teoría de Willis (13.4 contra 11.3–13.3 m). |
| Híbrido − σ: D_m −5.1 m en una hora del ciclo diurno; ν, κ_t en la base de la capa de mezcla | **(c)**; el punto abierto (umbral de 1 mm) **corregido en la tarea 9c** | Malla distinta (capas que colapsan en la base). Con el umbral relativo (1 cm aquí) RMS 0.72 → 0.48 m y la peor hora −5.1 → −2.85 m (otro día, con las capas delgadas ya fusionadas). |
| FORCED_OSCNONROTBBL: k, ε 3-5 % del rango a media corrida | **(c)** | En las fases de inversión del flujo, k pequeño; la amplitud y fase de v̄ coinciden a 0.4 % y 0°. |
| FORCED_NONROTBBL v̄ 0.7 % menor que CROCO | **(c)** | Cambia la discretización temporal del esfuerzo de fondo (σ de pyOCEAN: C_d \|u*\| linealizado sobre u + Δt G) — no investigado más; está dentro del 1 %. |

## Qué debe mejorar pyOCEAN

1. ✅ **Arrastre log con la rapidez completa** √(u² + v̄²) en cada cara (como CROCO y MOM6) en σ y ALE: hecho en la
   tarea 9b (`physics.bottom_friction.cross_velocity_at_faces`); el ángulo de la capa de Ekman de fondo pasó de 62.8° a
   76.0° (CROCO 75.8°). Cambia Lorenz/DOME/regional con rotación y z0. (clase a)
2. ✅ **Umbral de fusión de capas delgadas de la GLS**: hecho en la tarea 9c. h_thin = min(10 cm, max(1 mm, 1 % de
   D/N)) por columna (piso y tope = onemm y tencm de HYCOM, mxkprf.F90:1866-1878; la forma 0.1 √(κ₀ Δt) de MOM6,
   MOM_kappa_shear.F90:220-221, da < 3 mm en Lorenz y DOME) y las capas delgadas de superficie como capa fusionada
   propia (MOM_kappa_shear.F90:256-265). Ciclo diurno híbrido: RMS de D_m contra σ 0.72 → 0.48 m. (clase a)
3. **Costo de las columnas periódicas:** 10-25 × CROCO; un camino sin gradientes horizontales (o un halo mínimo) para
   columnas haría baratas las calibraciones y las columnas de MOM6 (tarea 9).
4. **Fuerza de cuerpo y flujo de calor de fondo como entradas del paso** (hoy en el arnés): las necesitan las columnas y
   cualquier caso idealizado con gradiente de presión impuesto.
5. **Diagnóstico de mezcla numérica de k y ε en el remapeo** (anti-objetivo 3): no existe todavía; aquí solo la cota
   indirecta de D_m.

## Columnas MOM6/CVMix (KPP, tarea 9; criterio 2 del spec)

Los cuatro casos KPP de `MOM6-examples/ocean_only/CVmix_SCM_tests` con la configuración exacta de MOM6
(`common_KPP/MOM_input` + el `KPP/MOM_override` de cada caso; decisión 9 del spec): 400 capas de 1 m, 2 × 2 reentrante,
Δt 1200 s, 5 días, f = 1e-4 s⁻¹, g = 9.80616, ρ₀ = 1000, EOS lineal dρ/dT = −0.255 (S = 35 uniforme y sin flujo de sal:
pyOCEAN corre sin salinidad, la densidad que ve KPP es la misma), T = max(4, 15 + 0.01 z) (isoterma 15 °C en
`skin_warming_wind`: `#override SCM_L2_DTDZ = 0.0`, confirmado en su `MOM_parameter_doc.all`), KPP con los valores por
defecto de CVMix e `INTERP_TYPE = "cubic"`, sin mezcla interior (KD = KV = 0), sin arrastre de fondo, sin onda corta.
**c_p = 3992.10322329649 J kg⁻¹ K⁻¹**, el `C_P` de este MOM_input (no el 3985 de CROCO ni el valor por defecto de MOM6):
así Q → F_T → B_f son los mismos números (B_f = ∓6.2638e-8 m² s⁻³ a ∓100 W m⁻², igual a `KPP_buoyFlux` de MOM6).
pyOCEAN en ALE z* (la coordenada de MOM6) y en σ (los mismos niveles en una columna plana). Nada se ajustó.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
validation/turbulence_columns/mom6/run_mom6.sh wind_only                  # ≈ 8 s por caso, 1 proceso
python -m validation.turbulence_columns.mom6_scm run --all                # 8 corridas, ≈ 3 min
python -m validation.turbulence_columns.mom6_scm compare [--series]
```

Salidas: `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/turbulence_columns/{mom6/<caso>/,pyocean_mom6/}` (nunca en el repo).
MOM6: `~/pyOCEAN_refs/MOM6/.testing/build/symmetric/MOM6` (git 0e9a561, CVMix fce4221), archivos de MOM6-examples sin
cambios (copiados al directorio de corrida).

**u\* con ráfaga (`mech_only`).** MOM6 usa u\* = ((|τ| + GUST_CONST)/ρ₀)^½ (`solo_driver/MOM_surface_forcing.F90:459`);
la ráfaga entra solo en u\*, no en el momento. `step()` de pyOCEAN no tiene ráfaga: el arnés suma gust/ρ₀ al u\*² que el
driver pasa a `KPPClosure.update` (`_install_gust`, solo esa instancia; `check_gust` falla si no llegó), como la fuerza de
cuerpo de la tarea 8. El modelo no cambia.

**Definiciones (iguales para los dos modelos).**
- h_OBL: la profundidad de la capa límite de KPP de cada paso (MOM6 `KPP_OBLdepth`, pyOCEAN `KPPClosure.obl_depth`),
  promediada en las ventanas de 2 h como hace el `diag_table` de MOM6 (`time_avg = .true.`, 6 pasos).
- MLD₀₀₃: el algoritmo de MOM6 `diagnoseMLDbyDensityDifference` (`MOM_diagnose_MLD.F90:34-247`, `HREF_FOR_MLD = 0`):
  profundidad, interpolada linealmente entre centros de capa, donde ρ − ρ(capa superior) llega a 0.03 kg m⁻³; el centro
  más profundo si no llega. Se calcula con `mld_003` sobre las fotos de T cada 2 h de ambos modelos (MOM6 `prog.nc`, con
  `MOM_IC.nc` en t = 0); entre paréntesis el `MLD_003` propio de MOM6 (promedio de ventana), que `mld_003` reproduce.
- "Δ final" = día 5 (última ventana); "máx |Δ|" = máximo sobre las salidas de los días 1-5.

| caso | coord | h_OBL MOM6 [m] | h_OBL pyOCEAN | Δ final | máx \|Δ\| d 1-5 | MLD MOM6 (diag) [m] | MLD pyOCEAN | Δ final | máx \|Δ\| d 1-5 | SST MOM6 / py [°C] | u sup. MOM6 / py [m/s] | ±5 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_only | z\* y σ | 27.81 | 27.81 | −0.0 % | 0.3 % | 28.33 (28.32) | 28.33 | −0.0 % | 0.2 % | 14.861 / 14.861 | 0.0505 / 0.0510 | **pasa** |
| cooling_only | z\* y σ | 50.70 | 50.70 | −0.0 % | 0.0 % | 58.96 (58.79) | 58.96 | +0.0 % | 0.0 % | 14.528 / 14.528 | 0 / 0 | **pasa** |
| mech_only | z\* y σ | 9.85 | 9.85 | +0.0 % | 0.0 % | 16.79 (16.78) | 16.79 | −0.0 % | 0.0 % | 14.950 / 14.950 | 0 / 0 | **pasa** |
| skin_warming_wind | z\* y σ | 26.69 | 26.69 | −0.0 % | 0.0 % | 26.69 (26.69) | 26.69 | +0.0 % | 0.3 % | 15.416 / 15.416 | 0.0685 / 0.0689 | (se reporta) |

Perfiles completos, cada 2 h durante 5 días: `cooling_only` y `mech_only` max |T_py − T_MOM6| = 4.8e-7 °C, el redondeo
del `float32` de `prog.nc` (packing 2): **igual a MOM6 a la precisión de la salida**. `wind_only`: max |ΔT| 6.8e-3 °C,
max |Δu| 1.7e-3 m/s de 0.13 (1.3 %), |Δv| 2.0e-3; `skin_warming_wind`: |ΔT| 9.2e-5 °C, |Δu| 7.6e-4 de 0.12. h_OBL y
MLD a ≤ 0.3 % en todo momento después del día 1. σ = z\* a 4e-12 (T), 2e-12 m/s (u), 8e-9 m (h_OBL). La columna 2 × 2 se
queda 1-D (dispersión 0).

**Criterio 2: cumplido** en `wind_only`, `cooling_only` y `mech_only`, en z\* y en σ (|Δ| ≤ 0.3 % en h_OBL y MLD, a 5 días
y en cualquier salida de los días 1-5). `skin_warming_wind` igual (0.0 % final, 0.3 % máx).

| diferencia | clase | evidencia |
|---|---|---|
| Casos con viento: u, v de la oscilación inercial ~1.3 % (wind_only), ~0.6 % (skin_warming_wind); T ≤ 6.8e-3 °C | **(c)** probable, no aislado | Solo aparece donde hay momento (los casos sin viento son iguales a la precisión de salida, así que KPP, el flujo de calor, la NLT y la difusión implícita coinciden). El esquema temporal del momento difiere: MOM6 `SPLIT = False` (dinámica no partida, predictor–corrector) contra AB3 + viscosidad implícita de pyOCEAN, con fΔt = 0.12. No se aisló con una oscilación inercial sin viscosidad. El efecto en h_OBL y MLD es ≤ 0.3 %. |
| Primera ventana de `skin_warming_wind`: h_OBL = 100.8 m (MOM6) | igual en pyOCEAN | Columna isoterma en reposo: Ri_b = 0 en toda la columna, h = fondo (399.5 m) en el primer paso, como CVMix. |

**Qué debe mejorar pyOCEAN (de esta tarea).**
1. **u\* con ráfaga como entrada del paso** (`GUST_CONST`; hoy en el arnés): la necesitan KPP y cualquier forzamiento
   con viento débil (u\* = 0 apaga la mezcla mecánica).
2. **Diagnósticos de KPP en el registro de variables** (`KPP_OBLdepth`, MLD₀₀₃, Ri_b, V_t², w_s) con promedio de ventana
   como MOM6: hoy el arnés lee `KPPClosure.obl_depth` cada paso.
3. **Costo de la columna:** 38-79 ms/paso (400 capas, 2 × 2 con halo periódico) contra ≈ 16 ms/paso de MOM6 (5.8-6.4 s por
   360 pasos, un proceso); KPP es NumPy (decisión de la tarea 7).
4. Aislar la diferencia de la oscilación inercial (AB3 contra el predictor–corrector de MOM6) con una prueba sin mezcla.

Tiempos de reloj (5 hilos Numba): wind_only z\* 28.4 s (79 ms/paso), σ 23.5 s (65); cooling_only 17.5 / 13.9 s;
mech_only 17.0 / 13.6 s; skin_warming_wind 28.0 / 23.4 s. MOM6: 5.8-6.4 s de "Total runtime" por caso (8-9 s de reloj
con el arranque de MPI).

## Archivos

`cases.py` (definición de los casos y `build`), `run_pyocean.py` (corridas, fotos horarias), `compare.py` (métricas y
tablas), `croco/build_croco.sh`, `croco/run_croco.sh`; `mom6_scm.py` (casos MOM6/CVMix, corridas, MLD₀₀₃,
comparación), `mom6/run_mom6.sh`; test `tests/test_turbulence_columns.py` (casos = CROCO, fuerza
de cuerpo, definición de D_m, Kato–Phillips 6 h a ±10 % de Price; casos MOM6 = MOM_override, definición de MLD₀₀₃).
